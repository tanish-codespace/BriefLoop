"""
Critic agent: must produce structured (score, feedback) the graph can route
on, and must never crash the pipeline if the model returns malformed JSON.
"""

from tests.conftest import FakeMessage
from agents.critic import critic_node


def test_critic_parses_valid_json(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.return_value = FakeMessage(
        content='{"score": 8.5, "feedback": "Tighten the timeline section."}'
    )
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: fake_llm)

    state = {**base_state, "draft": "draft text", "research_notes": "notes"}
    result = critic_node(state)

    assert result["critique_score"] == 8.5
    assert result["critique_feedback"] == "Tighten the timeline section."
    assert result["critique_history"] == ["Score 8.5/10: Tighten the timeline section."]
    assert result["agent_log"][-1]["agent"] == "Critic"


def test_critic_falls_back_on_malformed_json(monkeypatch, fake_llm, base_state):
    # Model ignored instructions and returned prose instead of JSON.
    fake_llm.invoke.return_value = FakeMessage(content="This draft looks great overall!")
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: fake_llm)

    state = {**base_state, "draft": "draft text", "research_notes": "notes"}
    result = critic_node(state)

    # Must not raise, and must still return a usable (score, feedback) pair
    # so route_after_critic has something to work with.
    assert isinstance(result["critique_score"], float)
    assert isinstance(result["critique_feedback"], str)
    assert result["critique_feedback"]  # non-empty


def test_critic_falls_back_on_missing_keys(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.return_value = FakeMessage(content='{"score": 6.0}')  # missing "feedback"
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: fake_llm)

    state = {**base_state, "draft": "draft text", "research_notes": "notes"}
    result = critic_node(state)

    assert isinstance(result["critique_score"], float)
    assert isinstance(result["critique_feedback"], str)


def test_critic_appends_to_existing_history(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.return_value = FakeMessage(content='{"score": 9.0, "feedback": "Good."}')
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: fake_llm)

    state = {**base_state, "draft": "d", "research_notes": "n", "critique_history": ["Score 5.0/10: earlier round"]}
    result = critic_node(state)

    assert result["critique_history"] == [
        "Score 5.0/10: earlier round",
        "Score 9.0/10: Good.",
    ]
