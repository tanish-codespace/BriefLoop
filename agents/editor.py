"""
Editor agent.

Job: final polish pass once the Critic approves (or the revision cap is
hit). Tightens language, fixes formatting, adds a one-line executive
summary at the top. Does NOT change substance.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm
from state import CampaignState

SYSTEM_PROMPT = """You are a copy editor for marketing plans. Polish the
draft's language and formatting - do not change its substance or add new
ideas. Add a one-line executive summary at the very top under a heading
"## Executive Summary". Keep the existing section structure below it."""


def editor_node(state: CampaignState) -> dict:
    llm = get_llm(temperature=0.3)
    draft = state.get("draft", "")

    msg = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Draft to polish:\n{draft}"),
    ])

    log_entry = {"agent": "Editor", "message": "Polished final draft and added executive summary."}

    return {
        "final_output": msg.content,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }
