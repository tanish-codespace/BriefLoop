"""
Budget & Channel-Fit Analyst agent.

Job: turn the brief (plus research notes) into a concrete budget constraint
and a channel allocation that actually fits it. This closes the gap the
README used to flag as "future work" - without it, the Writer was free to
recommend channels/tactics with no cost grounding, and the Critic had no way
to catch that.

If the brief states a number, we use it. If it doesn't, we infer a sane
default from campaign type/scale so the rest of the pipeline still has a
concrete constraint to design against (and we say so explicitly, so it's
never silently invented).
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm
from state import CampaignState

SYSTEM_PROMPT = """You are a marketing budget analyst. Given a campaign brief
and research notes, produce:
1. A stated or reasonably inferred budget constraint (if the brief gives a
   number, use it verbatim; otherwise infer a realistic range for the
   described campaign scale and say clearly that it's an assumption).
2. A channel allocation that fits that budget - which channels get the most
   spend and why, and which tactics from the research are NOT affordable at
   this budget and should be dropped or downsized.

Be concrete with numbers/percentages where possible. Keep it under 200 words.
Output plain text, no JSON."""

# Very light heuristic just to give the model a number to react to when the
# brief is silent on budget - keeps downstream output concrete instead of
# vague ("modest budget") for the common case where the user just forgot to
# mention it.
_MONEY_PATTERN = re.compile(r"\$\s?[\d,]+(?:\.\d+)?(?:\s?[kKmM])?")


def _extract_stated_budget(brief: str) -> str | None:
    match = _MONEY_PATTERN.search(brief)
    return match.group(0) if match else None


def budget_analyst_node(state: CampaignState) -> dict:
    llm = get_llm(temperature=0.3)
    brief = state["brief"]
    notes = state.get("research_notes", "")

    stated = _extract_stated_budget(brief)
    hint = (
        f"The brief explicitly states a budget: {stated}. Use this exact figure."
        if stated
        else "The brief does not state a budget. Infer a realistic range for "
        "this campaign's scale and say explicitly that it's an assumption."
    )

    msg = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Campaign brief: {brief}\n\n"
            f"Budget hint: {hint}\n\n"
            f"Research notes:\n{notes}"
        )),
    ])

    log_entry = {
        "agent": "Budget Analyst",
        "message": "Estimated budget constraint and channel allocation."
        if not stated else f"Used stated budget ({stated}) to allocate channels.",
    }

    return {
        "budget_notes": msg.content,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }
