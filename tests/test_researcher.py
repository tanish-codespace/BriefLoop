"""
Researcher agent: plans search queries, runs them through the (mocked)
search tool, then synthesizes results into notes. Two separate LLM calls
happen here, so the fake needs `side_effect` rather than a single
`return_value`.
"""

from tests.conftest import FakeMessage
from agents.researcher import researcher_node


def test_researcher_runs_planned_queries_and_synthesizes(monkeypatch, fake_llm, base_state):
    fake_llm.invoke.side_effect = [
        FakeMessage(content="college student fitness app market\ncompetitor fitness apps"),
        FakeMessage(content="Synthesized research summary [1][2]."),
    ]
    monkeypatch.setattr("agents.researcher.get_llm", lambda **kw: fake_llm)

    search_calls = []

    def fake_web_search(query, max_results=5):
        search_calls.append(query)
        return {
            "query": query,
            "results": [
                {"title": f"Result for {query}", "url": f"https://example.com/{len(search_calls)}", "snippet": "..."},
            ],
        }

    monkeypatch.setattr("agents.researcher.web_search", fake_web_search)

    result = researcher_node(base_state)

    assert search_calls == ["college student fitness app market", "competitor fitness apps"]
    assert result["research_notes"] == "Synthesized research summary [1][2]."
    assert result["sources"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert "Ran 2 searches" in result["agent_log"][-1]["message"]


def test_researcher_caps_queries_at_four(monkeypatch, fake_llm, base_state):
    six_queries = "\n".join(f"query {i}" for i in range(6))
    fake_llm.invoke.side_effect = [
        FakeMessage(content=six_queries),
        FakeMessage(content="notes"),
    ]
    monkeypatch.setattr("agents.researcher.get_llm", lambda **kw: fake_llm)
    monkeypatch.setattr(
        "agents.researcher.web_search",
        lambda query, max_results=5: {"query": query, "results": []},
    )

    result = researcher_node(base_state)

    assert "Ran 4 searches" in result["agent_log"][-1]["message"]
