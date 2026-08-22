"""
require_approval=True should pause the graph right before the Editor node
and resume cleanly from there - this is the mechanism the CLI's --review
flag and the Streamlit approval gate both rely on.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from graph import build_graph


def _fake_llm(constant_content=None, content_by_call=None):
    llm = MagicMock()
    if content_by_call is not None:
        llm.invoke.side_effect = [SimpleNamespace(content=c) for c in content_by_call]
    else:
        llm.invoke.return_value = SimpleNamespace(content=constant_content)
    return llm


def test_pauses_before_editor_and_resumes(monkeypatch):
    monkeypatch.setattr(
        "agents.researcher.web_search",
        lambda query, max_results=5: {"query": query, "results": []},
    )
    monkeypatch.setattr("agents.researcher.get_llm", lambda **kw: _fake_llm(
        content_by_call=["query one", "notes"]))
    monkeypatch.setattr("agents.budget_analyst.get_llm", lambda **kw: _fake_llm(
        constant_content="budget notes"))
    monkeypatch.setattr("agents.writer.get_llm", lambda **kw: _fake_llm(
        constant_content="draft plan"))
    monkeypatch.setattr("agents.critic.get_llm", lambda **kw: _fake_llm(
        constant_content='{"score": 9.0, "feedback": "Great."}'))
    monkeypatch.setattr("agents.editor.get_llm", lambda **kw: _fake_llm(
        constant_content="## Executive Summary\n\ndraft plan"))

    app = build_graph(require_approval=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "brief": "Plan a launch campaign.",
        "max_revisions": 2,
        "revision_count": 0,
        "critique_history": [],
        "agent_log": [],
    }

    paused = app.invoke(initial_state, config=config)

    # Interrupted before the Editor ran - critic output is present, final
    # output is not.
    assert paused["critique_score"] == 9.0
    assert "final_output" not in paused or not paused.get("final_output")
    assert [e["agent"] for e in paused["agent_log"]] == [
        "Researcher", "Budget Analyst", "Writer", "Critic",
    ]

    resumed = app.invoke(None, config=config)

    assert resumed["final_output"].startswith("## Executive Summary")
    assert [e["agent"] for e in resumed["agent_log"]] == [
        "Researcher", "Budget Analyst", "Writer", "Critic", "Editor",
    ]
