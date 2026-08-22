"""
Researcher agent.

Job: turn the raw campaign brief into a handful of targeted search
queries, run them, and synthesize the findings into notes the Writer
can actually use (not a dump of raw search results).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm
from tools import web_search
from state import CampaignState

SYSTEM_PROMPT = """You are a marketing research analyst. Given a campaign brief,
you produce a SHORT plan of 2-4 targeted web search queries that would surface
useful facts: market trends, competitor campaigns, target audience insights,
or relevant statistics. Return ONLY the queries, one per line, no numbering."""

SYNTHESIS_PROMPT = """You are a marketing research analyst. Below is a campaign
brief and a set of raw web search results. Write a concise research summary
(under 300 words) that the campaign writer can use directly: key facts,
audience insights, competitor angles, and any relevant statistics.
Cite sources inline like [1], [2] matching the numbered results."""


def researcher_node(state: CampaignState) -> dict:
    llm = get_llm(temperature=0.3)
    brief = state["brief"]

    # Step 1: ask the model to plan search queries
    plan_msg = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Campaign brief: {brief}"),
    ])
    queries = [q.strip("- ").strip() for q in plan_msg.content.strip().split("\n") if q.strip()]
    queries = queries[:4]  # cap it

    # Step 2: run the searches
    all_results = []
    sources = []
    for q in queries:
        result = web_search(q)
        all_results.append(result)
        sources.extend([r["url"] for r in result["results"]])

    # Step 3: synthesize into notes
    results_text = ""
    counter = 1
    for r in all_results:
        for item in r["results"]:
            results_text += f"[{counter}] {item['title']} ({item['url']})\n{item['snippet']}\n\n"
            counter += 1

    synth_msg = llm.invoke([
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=f"Brief: {brief}\n\nSearch results:\n{results_text}"),
    ])

    log_entry = {"agent": "Researcher", "message": f"Ran {len(queries)} searches, synthesized findings."}

    return {
        "research_notes": synth_msg.content,
        "sources": sources,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }
