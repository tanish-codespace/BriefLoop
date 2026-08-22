"""
Writer agent.

Job: produce (or revise) the campaign plan draft. On the first pass it
works from research notes. On later passes it incorporates the Critic's
feedback directly - this is the "self-correct" half of the feedback loop.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm
from state import CampaignState

SYSTEM_PROMPT = """You are a senior marketing strategist. Write a clear,
structured campaign plan with these sections: Objective, Target Audience,
Key Message, Channels & Tactics, Timeline (high-level), Success Metrics.
Be specific and grounded in the research notes provided - avoid generic
filler. The Channels & Tactics section must respect the budget notes
provided: only recommend tactics that fit the stated/inferred budget, and
reflect the suggested channel allocation. Keep it under 500 words."""

REVISION_PROMPT = """You are revising a campaign plan based on critic feedback.
Keep everything that already works. Directly address every point in the
feedback. Do not just add a paragraph acknowledging the feedback - actually
change the plan. Keep it under 500 words."""


def writer_node(state: CampaignState) -> dict:
    llm = get_llm(temperature=0.6)
    brief = state["brief"]
    notes = state.get("research_notes", "")
    budget_notes = state.get("budget_notes", "")
    revision_count = state.get("revision_count", 0)

    if revision_count == 0:
        # first draft
        msg = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Campaign brief: {brief}\n\n"
                f"Research notes:\n{notes}\n\n"
                f"Budget notes:\n{budget_notes}"
            )),
        ])
        log_msg = "Wrote first draft from research notes."
    else:
        # revision informed by critic feedback
        feedback = state.get("critique_feedback", "")
        prior_draft = state.get("draft", "")
        msg = llm.invoke([
            SystemMessage(content=REVISION_PROMPT),
            HumanMessage(content=(
                f"Campaign brief: {brief}\n\n"
                f"Budget notes:\n{budget_notes}\n\n"
                f"Previous draft:\n{prior_draft}\n\n"
                f"Critic feedback to address:\n{feedback}"
            )),
        ])
        log_msg = f"Revised draft (revision #{revision_count}) based on critic feedback."

    log_entry = {"agent": "Writer", "message": log_msg}

    return {
        "draft": msg.content,
        "revision_count": revision_count + 1,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }
