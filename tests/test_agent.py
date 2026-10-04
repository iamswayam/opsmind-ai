def test_investigation_loop_stops_at_step_cap(monkeypatch):
    import app.agent as agent

    call_count = 0

    def fake_generate_structured(prompt, system_instruction):
        nonlocal call_count
        call_count += 1
        return {
            "check": f"fake_check_{call_count}",
            "result": "unknown",
            "reason": "test double — not a real model response",
            "evidence_status": "insufficient",
            "missing_evidence": [],
            "next_action": "investigate",
        }

    monkeypatch.setattr(agent, "generate_structured", fake_generate_structured)

    state = agent.initial_state("test question", [], None)
    state["intent"] = "investigate"
    state["retrieved_chunks"] = [
        {"filename": "fake.pdf", "content": "irrelevant test content", "metadata": {}}
    ]

    safety_limit = 10
    iterations = 0
    next_action = None
    while iterations < safety_limit:
        state = agent.investigate_node(state)
        iterations += 1
        next_action = agent.route_after_investigate(state)
        if next_action != "investigate":
            break

    assert state["troubleshooting_step"] <= agent.MAX_INVESTIGATION_STEPS
    assert next_action == "escalate"
    assert len(state["investigation_steps"]) == state["troubleshooting_step"]
    assert iterations < safety_limit