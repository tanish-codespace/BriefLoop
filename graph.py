"""
This is the core orchestration: wires Researcher -> Budget Analyst -> Writer
-> Critic -> Editor into a LangGraph StateGraph, with a conditional edge
that sends the Critic's verdict either back to the Writer (revise) or
forward to the Editor (approve).

The loop is capped by max_revisions so a stubborn critic can't spin forever -
this is the kind of guardrail worth mentioning explicitly in an interview.

Human-in-the-loop: build_graph(require_approval=True) compiles the graph
with a checkpointer and an interrupt right before the Editor node. Callers
(CLI, Streamlit) run/stream up to that interrupt, inspect the approved draft
+ critic score, then resume with app.invoke(None, config) to let the Editor
run - or stop there and use the pre-Editor draft as-is. Without
require_approval the pipeline runs fully autonomously, same as before.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from state import CampaignState
from agents.researcher import researcher_node
from agents.budget_analyst import budget_analyst_node
from agents.writer import writer_node
from agents.critic import critic_node
from agents.editor import editor_node

SCORE_THRESHOLD = 7.5  # critic must score >= this to approve


def route_after_critic(state: CampaignState) -> str:
    """Decide whether to loop back to the writer or move on to the editor."""
    score = state.get("critique_score", 0)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)

    if score >= SCORE_THRESHOLD:
        return "editor"
    if revision_count >= max_revisions:
        return "editor"  # cap hit - ship what we have rather than loop forever
    return "writer"


def build_graph(require_approval: bool = False):
    graph = StateGraph(CampaignState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("budget_analyst", budget_analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("editor", editor_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "budget_analyst")
    graph.add_edge("budget_analyst", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "editor": "editor"},
    )
    graph.add_edge("editor", END)

    if require_approval:
        # Checkpointer is required for interrupt/resume to work - it's what
        # lets the graph pause mid-run and be resumed later from saved state.
        return graph.compile(checkpointer=MemorySaver(), interrupt_before=["editor"])
    return graph.compile()


def run_campaign(brief: str, max_revisions: int = 2):
    """Convenience entrypoint - runs the full pipeline and returns final state."""
    app = build_graph()
    initial_state: CampaignState = {
        "brief": brief,
        "max_revisions": max_revisions,
        "revision_count": 0,
        "critique_history": [],
        "agent_log": [],
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    import sys
    import uuid
    from dotenv import load_dotenv
    load_dotenv()

    args = sys.argv[1:]
    review = "--review" in args
    args = [a for a in args if a != "--review"]

    brief = " ".join(args) or "Plan a launch campaign for a new budget fitness tracking app targeting college students."

    if not review:
        result = run_campaign(brief)
    else:
        # Human-in-the-loop CLI demo: pause right before the Editor runs.
        app = build_graph(require_approval=True)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state: CampaignState = {
            "brief": brief,
            "max_revisions": 2,
            "revision_count": 0,
            "critique_history": [],
            "agent_log": [],
        }
        result = app.invoke(initial_state, config=config)

        print("\n=== PAUSED FOR HUMAN APPROVAL (pre-Editor) ===")
        print(f"Critic score: {result.get('critique_score')}/10")
        print(f"\nDraft:\n{result.get('draft')}\n")
        answer = input("Approve and run final Editor polish? [y/N] ").strip().lower()
        if answer == "y":
            result = app.invoke(None, config=config)
        else:
            print("\nRejected - shipping the pre-Editor draft as final output.")
            result["final_output"] = result.get("draft", "")

    print("\n=== AGENT LOG ===")
    for entry in result["agent_log"]:
        print(f"[{entry['agent']}] {entry['message']}")

    print("\n=== FINAL CAMPAIGN PLAN ===\n")
    print(result["final_output"])
