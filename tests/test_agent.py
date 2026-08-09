"""
Standalone test for the investigate loop's step-cap enforcement.

Runs entirely offline — no real Gemini API calls, no Postgres connection.
Monkeypatches generate_structured so the fake model ALWAYS claims it wants
to keep investigating, then verifies the code-level guard in investigate_node
forces escalation at MAX_INVESTIGATION_STEPS regardless of what the model says.

This proves the mechanism deterministically, rather than hoping a real
question happens to need exactly 3 steps before you can observe it — and it
costs zero quota.

Run inside the container (needs the same environment/dependencies as the app):
    docker compose exec api python -m app.test_investigate_loop
"""

import app.agent as agent

# ---- Fake the model: it NEVER wants to stop on its own ----
call_count = 0


def fake_generate_structured(prompt, system_instruction):
    global call_count
    call_count += 1
    return {
        "check": f"fake_check_{call_count}",
        "result": "unknown",
        "reason": "test double — not a real model response",
        "evidence_status": "insufficient",
        "missing_evidence": [],
        "next_action": "investigate",  # always asks to continue, no matter what
    }


agent.generate_structured = fake_generate_structured

# ---- Drive the loop manually, the same way the graph's conditional edge would ----
state = agent.initial_state("test question", [], None)
state["intent"] = "investigate"
state["retrieved_chunks"] = [{"filename": "fake.pdf", "content": "irrelevant test content", "metadata": {}}]

SAFETY_LIMIT = 10  # guards the TEST ITSELF in case of an actual infinite-loop bug
iterations = 0
next_action = None

while iterations < SAFETY_LIMIT:
    state = agent.investigate_node(state)
    iterations += 1
    next_action = agent.route_after_investigate(state)
    print(f"Step {state['troubleshooting_step']}: model asked for 'investigate', "
          f"code enforced next_action = '{next_action}'")

    if next_action != "investigate":
        break

# ---- Assertions ----
assert state["troubleshooting_step"] <= agent.MAX_INVESTIGATION_STEPS, (
    f"FAIL: loop ran {state['troubleshooting_step']} steps, exceeding "
    f"MAX_INVESTIGATION_STEPS={agent.MAX_INVESTIGATION_STEPS} — the cap is not being enforced."
)
assert next_action == "escalate", (
    f"FAIL: expected the cap to force 'escalate' once the model kept saying 'investigate', "
    f"but got '{next_action}' instead."
)
assert len(state["investigation_steps"]) == state["troubleshooting_step"], (
    "FAIL: investigation_steps length should exactly match troubleshooting_step — "
    "one recorded check per step, no silent extra checks slipping in."
)
assert iterations < SAFETY_LIMIT, (
    "FAIL: hit the test's own safety limit — this means the real cap never fired at all."
)

print(f"\nPASS — loop stopped at step {state['troubleshooting_step']} "
      f"(cap = {agent.MAX_INVESTIGATION_STEPS}), forced to 'escalate' "
      f"even though the fake model always requested 'investigate'.")
print(f"Fake model was called {call_count} times, zero real API calls made.")