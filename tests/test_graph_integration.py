"""
End-to-end graph tests with every LLM/search call mocked. These exercise
the actual wiring (StateGraph edges + conditional routing), not just each
node in isolation - the thing most likely to break silently when someone
adds or reorders a node.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graph import build_graph, run_campaign


def _fake_llm_for(agent_name, content_by_call=None, constant_content=None):
    """Builds a MagicMock LLM whose .invoke() returns canned content, either
    a fixed string every time or a list consumed one-per-call."""
    llm = MagicMock()
    if content_by_call is not None:
        llm.invoke.side_effect = [SimpleNamespace(content=c) for c in content_by_call]
    else:
        llm.invoke.return_value = SimpleNamespace(content=constant_content)
    return llm


@pytest.fixture(autouse=True)
def mock_search(monkeypatch):
    monkeypatch.setattr(
        "agents.researcher.web_search",
        lambda query, max_results=5: {
            "query": query,
            "results": [{"title": "t", "url": "https://example.com/x", "snippet": "s"}],
        },
    )


def test_happy_path_no_revision_needed(monkeypatch):
    """Critic approves on the first draft - graph should go straight to the
    editor without looping back to the writer."""
    monkeypatch.setattr("agents.researcher.get_llm", lambda **kw: _fake_llm_for(
        "researcher", content_by_call=["query one", "research notes summary"]))
    monkeypatch.setattr("agents.budget_analyst.get_llm", lambda **kw: _fake_llm_for(
        "budget", constant_content="budget allocation notes"))
    monkeypatch.setattr("agents.writer.get_llm", lambda **kw: _fake_llm_for(
        "writer", constant_content="first draft plan"))
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: _fake_llm_for(
        "critic", constant_content='{"score": 9.0, "feedback": "Great work."}'))
    monkeypatch.setattr("agents.editor.get_llm", lambda **kw: _fake_llm_for(
        "editor", constant_content="## Executive Summary\n\nfirst draft plan"))

    result = run_campaign("Plan a launch campaign for a budget fitness app.", max_revisions=2)

    assert result["revision_count"] == 1  # writer only ran once
    assert result["critique_score"] == 9.0
    assert result["final_output"].startswith("## Executive Summary")
    assert [e["agent"] for e in result["agent_log"]] == [
        "Researcher", "Budget Analyst", "Writer", "Critic", "Editor",
    ]


def test_loop_back_to_writer_then_approves(monkeypatch):
    """Critic rejects the first draft, approves the second - graph should
    visit Writer and Critic twice each, then proceed to Editor."""
    # Writer and Critic each get invoked (get_llm() called) once per node
    # execution, and each execution visits the node again on the loop back -
    # so the SAME mock instance (with its side_effect list) must be reused
    # across calls to get_llm(), not rebuilt fresh each time.
    researcher_llm = _fake_llm_for("researcher", content_by_call=["query one", "research notes summary"])
    budget_llm = _fake_llm_for("budget", constant_content="budget allocation notes")
    writer_llm = _fake_llm_for("writer", content_by_call=["first draft plan", "revised draft plan"])
    critic_llm = _fake_llm_for("critic", content_by_call=[
        '{"score": 4.0, "feedback": "Audience is too vague."}',
        '{"score": 8.0, "feedback": "Much better."}',
    ])
    editor_llm = _fake_llm_for("editor", constant_content="## Executive Summary\n\nrevised draft plan")

    monkeypatch.setattr("agents.researcher.get_llm", lambda **kw: researcher_llm)
    monkeypatch.setattr("agents.budget_analyst.get_llm", lambda **kw: budget_llm)
    monkeypatch.setattr("agents.writer.get_llm", lambda **kw: writer_llm)
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: critic_llm)
    monkeypatch.setattr("agents.editor.get_llm", lambda **kw: editor_llm)

    result = run_campaign("Plan a launch campaign for a budget fitness app.", max_revisions=2)

    assert result["revision_count"] == 2
    assert result["critique_score"] == 8.0
    assert result["critique_history"] == [
        "Score 4.0/10: Audience is too vague.",
        "Score 8.0/10: Much better.",
    ]
    agent_sequence = [e["agent"] for e in result["agent_log"]]
    assert agent_sequence == [
        "Researcher", "Budget Analyst", "Writer", "Critic", "Writer", "Critic", "Editor",
    ]


def test_revision_cap_forces_editor_even_if_score_stays_low(monkeypatch):
    monkeypatch.setattr("agents.researcher.get_llm", lambda **kw: _fake_llm_for(
        "researcher", content_by_call=["query one", "research notes summary"]))
    monkeypatch.setattr("agents.budget_analyst.get_llm", lambda **kw: _fake_llm_for(
        "budget", constant_content="budget allocation notes"))
    monkeypatch.setattr("agents.writer.get_llm", lambda **kw: _fake_llm_for(
        "writer", constant_content="draft plan"))
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: _fake_llm_for(
        "critic", constant_content='{"score": 2.0, "feedback": "Still too generic."}'))
    monkeypatch.setattr("agents.editor.get_llm", lambda **kw: _fake_llm_for(
        "editor", constant_content="## Executive Summary\n\ndraft plan"))

    result = run_campaign("Plan a launch campaign for a budget fitness app.", max_revisions=1)

    # max_revisions=1: writer runs for round 1, critic rejects, cap is hit
    # (revision_count == max_revisions) so it's forced through to editor
    # without a second writer pass.
    assert result["revision_count"] == 1
    assert result["final_output"].startswith("## Executive Summary")


def test_build_graph_has_expected_nodes():
    app = build_graph()
    node_names = set(app.get_graph().nodes.keys())
    assert {"researcher", "budget_analyst", "writer", "critic", "editor"} <= node_names
