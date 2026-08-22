"""
Writer agent: first pass drafts from research/budget notes; on revision it
must switch prompts and actually incorporate the critic's feedback, and
revision_count must increment exactly once per call (route_after_critic and
the max_revisions cap both depend on this being reliable).
"""

from tests.conftest import FakeMessage
from agents.writer import writer_node


def test_first_draft_uses_system_prompt_and_sets_revision_count_to_1(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.return_value = FakeMessage(content="Objective: ...")
    monkeypatch.setattr("agents.writer.get_llm", lambda **kw: fake_llm)

    state = {**base_state, "research_notes": "notes", "budget_notes": "budget"}
    result = writer_node(state)

    assert result["draft"] == "Objective: ..."
    assert result["revision_count"] == 1
    assert "first draft" in result["agent_log"][-1]["message"].lower()

    # first draft: no prior critic feedback should be referenced
    sent_messages = fake_llm.invoke.call_args[0][0]
    combined = " ".join(m.content for m in sent_messages)
    assert "notes" in combined
    assert "budget" in combined


def test_revision_incorporates_feedback_and_increments_count(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.return_value = FakeMessage(content="Revised plan")
    monkeypatch.setattr("agents.writer.get_llm", lambda **kw: fake_llm)

    state = {
        **base_state,
        "revision_count": 1,
        "draft": "old draft",
        "critique_feedback": "Be more specific about the audience.",
        "budget_notes": "budget",
    }
    result = writer_node(state)

    assert result["draft"] == "Revised plan"
    assert result["revision_count"] == 2
    assert "revision #1" in result["agent_log"][-1]["message"].lower()

    sent_messages = fake_llm.invoke.call_args[0][0]
    combined = " ".join(m.content for m in sent_messages)
    assert "Be more specific about the audience." in combined
    assert "old draft" in combined
