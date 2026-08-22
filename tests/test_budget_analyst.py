"""
Budget Analyst agent: should pass a stated dollar figure straight through
to the model as a hard constraint, and should flag clearly when it's
inferring a budget because the brief didn't state one.
"""

from tests.conftest import FakeMessage
from agents.budget_analyst import budget_analyst_node, _extract_stated_budget


def test_extracts_stated_dollar_amount():
    assert _extract_stated_budget("Plan a campaign with a $5,000 budget.") == "$5,000"
    assert _extract_stated_budget("We have $2k to spend this quarter.") == "$2k"
    assert _extract_stated_budget("No budget mentioned here.") is None


def test_uses_stated_budget_in_prompt(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.return_value = FakeMessage(content="Allocate $3k to paid social...")
    monkeypatch.setattr("agents.budget_analyst.get_llm", lambda **kw: fake_llm)

    state = {**base_state, "brief": "Launch campaign, budget is $3,000.", "research_notes": "notes"}
    result = budget_analyst_node(state)

    assert result["budget_notes"] == "Allocate $3k to paid social..."
    sent_messages = fake_llm.invoke.call_args[0][0]
    combined = " ".join(m.content for m in sent_messages)
    assert "$3,000" in combined
    assert "explicitly states a budget" in combined
    assert "Used stated budget" in result["agent_log"][-1]["message"]


def test_infers_budget_when_not_stated(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.return_value = FakeMessage(content="Assuming a lean budget...")
    monkeypatch.setattr("agents.budget_analyst.get_llm", lambda **kw: fake_llm)

    state = {**base_state, "brief": "Launch campaign for a new app.", "research_notes": "notes"}
    result = budget_analyst_node(state)

    sent_messages = fake_llm.invoke.call_args[0][0]
    combined = " ".join(m.content for m in sent_messages)
    assert "does not state a budget" in combined
    assert "Estimated budget constraint" in result["agent_log"][-1]["message"]
