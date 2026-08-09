"""
OpsMind's agentic decision layer.

This is deliberately NOT a linear retrieve-then-generate pipeline like /chat.
A question is triaged into one of three intents, each with a different
execution path:

  - direct:      a specific, answerable question -> retrieve -> answer
  - clarify:     too vague to usefully act on -> ask a clarifying question
  - investigate: a diagnostic/troubleshooting question -> a bounded loop that
                 gathers evidence, evaluates one diagnostic check at a time,
                 remembers what it already checked, and either resolves,
                 asks for specific missing evidence, or escalates.

Design principles this file holds itself to (agreed before writing any code):
  - Nodes stay "boring": read state, do one thing, update state, return it.
    All the branching logic lives in the conditional edge functions, not
    hidden inside a node body.
  - The investigate loop is generic, not hardcoded to any one domain. It asks
    the model to pick the next relevant diagnostic check FROM THE RETRIEVED
    CONTEXT, rather than the graph having fixed nodes like "check_radius" or
    "check_acceptance_window". Domain-specific checks come from whatever SOP
    content gets retrieved — the vendor-dispatch scenario is a test case for
    this mechanism, not something baked into the graph itself.
  - evidence_status is a Literal ("sufficient"/"insufficient"/"conflicting"),
    not a numeric confidence score. There's no evaluation data to calibrate
    a float against, so a fake-precision number would be worse than an
    honest categorical judgment.
  - One investigation iteration = ONE combined LLM call that both picks the
    next check AND evaluates it. This is what actually enforces the call
    budget rather than just hoping the model behaves — a separate
    choose/evaluate pair would double the worst-case cost per step.
"""

from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph, END

from app.db import get_connection
from app.gemini_client import embed_text, generate_answer, generate_structured
from app.retrieval import retrieve_chunks, build_sources, build_context_images, DISTANCE_THRESHOLD

MAX_INVESTIGATION_STEPS = 3
# One "step" = one diagnostic check evaluated against available evidence and
# recorded in investigation_steps. A node is not allowed to silently perform
# multiple checks inside a single step — that would make this bound meaningless.


class AgentState(TypedDict):
    question: str
    history: list[dict]
    document_ids: Optional[list[int]]

    intent: Literal["direct", "clarify", "investigate"]
    next_action: str
    clarifying_question: Optional[str]

    retrieved_chunks: list[dict]
    retrieval_confidence: Optional[float]

    evidence_status: Optional[Literal["sufficient", "insufficient", "conflicting"]]
    missing_evidence: list[str]

    investigation_steps: list[dict]   # [{"check": str, "result": str, "reason": str}]
    troubleshooting_step: int

    resolved: bool
    answer: str
    sources: list[dict]


# ---------- Nodes ----------

def triage_node(state: AgentState) -> AgentState:
    """Single structured-output call: classifies intent and, if the question
    is ambiguous, produces the clarifying question in the SAME call — so the
    clarify path costs exactly 1 LLM call total, not 2."""
    prompt = (
        f"Conversation history: {state['history']}\n"
        f"Question: {state['question']}\n\n"
        "Classify this ops-support question and respond with ONLY a JSON object:\n"
        '{"intent": "direct" | "clarify" | "investigate", '
        '"clarifying_question": "<question to ask, only if intent is clarify, else null>"}\n\n'
        '- "direct": a specific, answerable question referencing a policy, SOP, or fact '
        "that a single document lookup can answer.\n"
        '- "clarify": missing key details (system, time period, case ID, etc.) needed '
        "before retrieval or investigation would be useful.\n"
        '- "investigate": a diagnostic/troubleshooting question about why something '
        "failed, requiring gathering evidence and reasoning about likely causes."
    )
    system_instruction = (
        "You are a triage classifier for an internal ops support system. "
        "Respond with strict JSON matching the described schema, nothing else."
    )
    result = generate_structured(prompt, system_instruction)

    state["intent"] = result.get("intent", "direct")
    state["clarifying_question"] = result.get("clarifying_question")
    state["next_action"] = "clarify" if state["intent"] == "clarify" else "retrieve"
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Same retrieval logic as /chat — no reason to reinvent this for the agent."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            question_embedding = embed_text(state["question"])
            retrieved = retrieve_chunks(cur, question_embedding, state.get("document_ids"))
    finally:
        conn.close()

    state["retrieved_chunks"] = retrieved
    state["retrieval_confidence"] = retrieved[0]["distance"] if retrieved else None
    return state


def answer_node(state: AgentState) -> AgentState:
    """Direct-intent path: same confidence gate as /chat before generating."""
    retrieved = state["retrieved_chunks"]
    if not retrieved or state["retrieval_confidence"] is None or state["retrieval_confidence"] > DISTANCE_THRESHOLD:
        state["answer"] = (
            "I couldn't find anything in your uploaded documents that addresses this. "
            "Try rephrasing, or attach a specific document if you know which one it's in."
        )
        state["sources"] = []
    else:
        context_chunks = [f"[{r['filename']}] {r['content']}" for r in retrieved]
        state["answer"] = generate_answer(state["question"], context_chunks, state["history"],
                                           build_context_images(retrieved))
        state["sources"] = build_sources(retrieved)
    state["resolved"] = True
    return state


def clarify_node(state: AgentState) -> AgentState:
    """No retrieval spent here — that's the point. Asking first saves an
    embedding + generation call on a question we can't usefully act on yet."""
    state["answer"] = state.get("clarifying_question") or "Could you clarify what you're asking about?"
    state["sources"] = []
    state["resolved"] = False
    return state


def investigate_node(state: AgentState) -> AgentState:
    """One investigation iteration = one combined LLM call that picks the next
    diagnostic check AND evaluates it against retrieved evidence, in the same
    response. The check itself is NOT hardcoded — the model chooses it based
    on what's actually in the retrieved SOP/incident content."""
    step_num = state.get("troubleshooting_step", 0) + 1
    state["troubleshooting_step"] = step_num

    context_chunks = [f"[{r['filename']}] {r['content']}" for r in state.get("retrieved_chunks", [])]
    already_checked = state.get("investigation_steps", [])

    prompt = (
        f"You are troubleshooting an ops incident.\n\n"
        f"Question: {state['question']}\n\n"
        f"Retrieved SOP/incident context:\n"
        f"{chr(10).join(context_chunks) if context_chunks else '(none retrieved)'}\n\n"
        f"Already checked this investigation (do not repeat these):\n"
        f"{already_checked if already_checked else '(nothing checked yet)'}\n\n"
        f"This is step {step_num} of a maximum {MAX_INVESTIGATION_STEPS}.\n\n"
        "Pick ONE diagnostic check not already covered above, relevant to the retrieved "
        "context, evaluate whether the evidence confirms it, rules it out, or is silent "
        "on it, and respond with ONLY a JSON object:\n"
        '{"check": "<short name of the check>", '
        '"result": "confirmed" | "ruled_out" | "unknown", '
        '"reason": "<brief evidence-based explanation>", '
        '"evidence_status": "sufficient" | "insufficient" | "conflicting", '
        '"missing_evidence": ["<specific missing fact, if any>"], '
        '"next_action": "answer" | "request_evidence" | "investigate" | "escalate"}\n\n'
        'Use "answer" only if evidence_status is "sufficient" and you can state a root cause.\n'
        '"request_evidence" if one specific missing fact would let you resolve this.\n'
        '"investigate" only if a genuinely different, relevant check remains untried.\n'
        f'"escalate" if no relevant checks remain, or this is step {MAX_INVESTIGATION_STEPS}.'
    )
    system_instruction = (
        "You are a rigorous ops troubleshooting assistant. Respond with strict JSON only."
    )
    result = generate_structured(prompt, system_instruction)

    state.setdefault("investigation_steps", []).append({
        "check": result.get("check", "unknown"),
        "result": result.get("result", "unknown"),
        "reason": result.get("reason", ""),
    })
    state["evidence_status"] = result.get("evidence_status")
    state["missing_evidence"] = result.get("missing_evidence", [])

    next_action = result.get("next_action", "escalate")
    # Enforce the bound in code, not just in the prompt — never trust a model
    # to self-limit a loop it's inside of.
    if step_num >= MAX_INVESTIGATION_STEPS and next_action == "investigate":
        next_action = "escalate"
    state["next_action"] = next_action
    return state


def investigate_answer_node(state: AgentState) -> AgentState:
    """Reached when the investigate loop decides it has sufficient evidence.
    Generates the final answer citing both the retrieved context and what
    was checked during investigation, so the answer explains its reasoning."""
    steps = state.get("investigation_steps", [])
    checked_summary = "; ".join(f"{s['check']}: {s['result']}" for s in steps)
    context_chunks = [f"[{r['filename']}] {r['content']}" for r in state.get("retrieved_chunks", [])]
    context_chunks.append(f"Investigation findings so far: {checked_summary}")

    state["answer"] = generate_answer(state["question"], context_chunks, state["history"],
                                       build_context_images(state.get("retrieved_chunks", [])))
    state["sources"] = build_sources(state.get("retrieved_chunks", []))
    state["resolved"] = True
    return state


def request_evidence_node(state: AgentState) -> AgentState:
    """Investigation found a specific gap it can't reason past — ask for it
    explicitly rather than guessing or looping pointlessly."""
    missing = state.get("missing_evidence") or ["more specific details about the incident"]
    state["answer"] = "I need more information to diagnose this. Specifically: " + "; ".join(missing)
    state["sources"] = build_sources(state.get("retrieved_chunks", []))
    state["resolved"] = False
    return state


def escalate_node(state: AgentState) -> AgentState:
    """Hit MAX_INVESTIGATION_STEPS or ran out of relevant checks. The escalation
    message is built FROM investigation_steps, so it's a real, specific summary
    of what was actually tried — not a generic "I don't know"."""
    steps = state.get("investigation_steps", [])
    checked = ", ".join(s["check"] for s in steps) or "no diagnostic checks"
    missing = state.get("missing_evidence") or []

    state["answer"] = (
        f"I checked {checked} but couldn't confidently resolve this. "
        + (f"Missing: {'; '.join(missing)}. " if missing else "")
        + "This needs a human to look at it — escalating."
    )
    state["sources"] = build_sources(state.get("retrieved_chunks", []))
    state["resolved"] = False
    return state


# ---------- Conditional edges (routing lives here, not inside nodes) ----------

def route_after_triage(state: AgentState) -> str:
    return state["next_action"]  # "clarify" | "retrieve"


def route_after_retrieve(state: AgentState) -> str:
    return "investigate" if state["intent"] == "investigate" else "answer"


def route_after_investigate(state: AgentState) -> str:
    return state["next_action"]  # "answer" | "request_evidence" | "investigate" | "escalate"


# ---------- Graph assembly ----------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("direct_answer", answer_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("investigate_answer", investigate_answer_node)
    graph.add_node("request_evidence", request_evidence_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("triage")

    graph.add_conditional_edges("triage", route_after_triage, {
        "clarify": "clarify",
        "retrieve": "retrieve",
    })

    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "answer": "direct_answer",
        "investigate": "investigate",
    })

    graph.add_conditional_edges("investigate", route_after_investigate, {
        "answer": "investigate_answer",
        "request_evidence": "request_evidence",
        "investigate": "investigate",   # the bounded loop
        "escalate": "escalate",
    })

    graph.add_edge("clarify", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("investigate_answer", END)
    graph.add_edge("request_evidence", END)
    graph.add_edge("escalate", END)

    return graph.compile()


def initial_state(question: str, history: list[dict], document_ids: Optional[list[int]]) -> AgentState:
    """Build a fresh AgentState for a new turn. Kept as one function so every
    caller (the /chat/agent endpoint, tests, etc.) constructs state identically."""
    return {
        "question": question,
        "history": history,
        "document_ids": document_ids,
        "intent": "direct",
        "next_action": "retrieve",
        "clarifying_question": None,
        "retrieved_chunks": [],
        "retrieval_confidence": None,
        "evidence_status": None,
        "missing_evidence": [],
        "investigation_steps": [],
        "troubleshooting_step": 0,
        "resolved": False,
        "answer": "",
        "sources": [],
    }
