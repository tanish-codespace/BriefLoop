"""
Critic agent.

Job: score the draft against a fixed rubric and give SPECIFIC, actionable
feedback. Returns structured JSON so the graph can programmatically decide
whether to loop back to the Writer or move on to the Editor.
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm
from state import CampaignState

SYSTEM_PROMPT = """You are a skeptical marketing director reviewing a campaign
plan draft. Score it 0-10 against this rubric:
- Clarity of objective (is it specific and measurable?)
- Audience specificity (is the target audience concrete, not "everyone"?)
- Channel/tactic feasibility (are the tactics realistic and on-brief?)
- Grounded in research (does it use the research notes, not generic advice?)
- Budget feasibility (does it respect the budget notes' constraint and
  allocation, not recommend tactics that were flagged as unaffordable?)

Respond ONLY with valid JSON in this exact shape, no other text:
{"score": <float 0-10>, "feedback": "<2-4 specific, actionable sentences>"}"""


def critic_node(state: CampaignState) -> dict:
    llm = get_llm(temperature=0.2)
    draft = state.get("draft", "")
    notes = state.get("research_notes", "")
    budget_notes = state.get("budget_notes", "")

    msg = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Research notes:\n{notes}\n\n"
            f"Budget notes:\n{budget_notes}\n\n"
            f"Draft to review:\n{draft}"
        )),
    ])

    try:
        parsed = json.loads(msg.content.strip())
        score = float(parsed["score"])
        feedback = parsed["feedback"]
    except (json.JSONDecodeError, KeyError, ValueError):
        # fallback if the model doesn't return clean JSON - don't crash the graph
        score = 7.0
        feedback = "Could not parse structured critique; proceeding with current draft."

    history = state.get("critique_history", [])
    history.append(f"Score {score}/10: {feedback}")

    log_entry = {"agent": "Critic", "message": f"Scored {score}/10 - {feedback}"}

    return {
        "critique_score": score,
        "critique_feedback": feedback,
        "critique_history": history,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }
