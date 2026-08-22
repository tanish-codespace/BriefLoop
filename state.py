"""
Shared state that flows through every agent in the graph.
LangGraph passes this dict-like object between nodes; each agent
reads what it needs and returns updates that get merged in.
"""

from typing import TypedDict, List


class CampaignState(TypedDict, total=False):
    # ---- input ----
    brief: str                     # the user's campaign request, e.g. "plan a launch campaign for a fitness app"

    # ---- researcher output ----
    research_notes: str            # synthesized findings
    sources: List[str]             # URLs used

    # ---- budget analyst output ----
    budget_notes: str              # budget constraint + channel allocation

    # ---- writer output ----
    draft: str                     # current draft of the campaign plan
    revision_count: int            # how many times the writer has revised

    # ---- critic output ----
    critique_score: float          # 0-10 rubric score
    critique_feedback: str         # specific, actionable feedback
    critique_history: List[str]    # log of all critiques (for the demo UI)

    # ---- editor output ----
    final_output: str              # polished final campaign plan

    # ---- control flow ----
    max_revisions: int             # cap so the loop can't spiral
    agent_log: List[dict]          # {"agent": ..., "message": ...} for UI display
