"""
Streamlit front end - "ops console" styling.

Design rationale: this app IS a pipeline (Researcher -> Writer -> Critic ->
Editor, with a possible loop back), so the UI leans into that directly -
a live pipeline tracker with meaningful status colors (running / approved /
revising) rather than a generic chat skin. Final output is organized into
tabs instead of stacked expanders so it reads like a report, not a wall
of text.
"""

import uuid

import streamlit as st
from dotenv import load_dotenv

from graph import build_graph
from state import CampaignState
from formatting import escape_markdown_dollars as md

load_dotenv()

st.set_page_config(
    page_title="Campaign Planner — Multi-Agent Pipeline",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0B1220;
    --surface: #131B2E;
    --border: #232E45;
    --text: #E8ECF4;
    --muted: #8A94AC;
    --accent: #5B8DEF;
    --success: #34D399;
    --warning: #F5A623;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

.console-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.console-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: -0.6rem;
    margin-bottom: 1.6rem;
}

/* Pipeline tracker */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.2rem 0 1.8rem 0;
    padding: 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
}

.stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
}

.stage-dot {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    border: 2px solid var(--border);
    background: var(--bg);
    color: var(--muted);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    z-index: 2;
}

.stage-dot.done { border-color: var(--success); background: var(--success); color: #0B1220; }
.stage-dot.active { border-color: var(--accent); background: var(--accent); color: #0B1220; animation: pulse 1.4s infinite; }
.stage-dot.revising { border-color: var(--warning); background: var(--warning); color: #0B1220; }

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(91,141,239,0.5); }
    70% { box-shadow: 0 0 0 8px rgba(91,141,239,0); }
    100% { box-shadow: 0 0 0 0 rgba(91,141,239,0); }
}

.stage-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 0.5rem;
}

.stage-sub {
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.1rem;
    text-align: center;
}

.stage-line {
    flex: 1;
    height: 2px;
    background: var(--border);
    margin-top: -20px;
}

.stage-line.done { background: var(--success); }

.loop-badge {
    font-size: 0.65rem;
    color: var(--warning);
    background: rgba(245,166,35,0.12);
    border: 1px solid var(--warning);
    border-radius: 20px;
    padding: 1px 8px;
    margin-top: 0.3rem;
    font-family: 'Space Grotesk', sans-serif;
}

/* Agent log cards */
.log-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
}

.log-agent {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    color: var(--text);
}

.log-msg {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.15rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
}
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--text); }
.metric-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="console-eyebrow">Multi-Agent Pipeline</div>', unsafe_allow_html=True)
st.title("Campaign Planner")
st.markdown(
    '<div class="console-subtitle">Researcher → Writer → Critic → Editor — '
    'with a real critic-to-writer feedback loop, not a scripted chain.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="console-eyebrow">Settings</div>', unsafe_allow_html=True)
    max_revisions = st.slider("Max writer revisions", 0, 4, 2)
    require_approval = st.checkbox(
        "Require human approval before final polish",
        value=False,
        help="Pauses the pipeline right before the Editor runs, so you can "
        "review the Critic-approved draft and either approve it or ship the "
        "unpolished draft instead.",
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline logic**\n\n"
        "The Critic scores each draft 0–10. Below **7.5**, it's sent back "
        "to the Writer with specific feedback. This can repeat up to the "
        "revision cap before being forced through to the Editor."
    )

AGENTS = [
    ("Researcher", "🔍", "web search"),
    ("Budget Analyst", "💰", "budget fit"),
    ("Writer", "✍️", "drafting"),
    ("Critic", "🧐", "scoring"),
    ("Editor", "✨", "polish"),
]

def render_pipeline(completed_agents, active_agent, revision_round):
    """Builds the HTML pipeline tracker reflecting current progress."""
    html = '<div class="pipeline">'
    for i, (name, icon, sub) in enumerate(AGENTS):
        if name in completed_agents and name != active_agent:
            dot_class, dot_content = "done", "✓"
        elif name == active_agent:
            dot_class, dot_content = ("revising" if revision_round > 1 and name in ("Writer", "Critic") else "active"), icon
        else:
            dot_class, dot_content = "", icon

        loop_badge = ""
        if name in ("Writer", "Critic") and revision_round > 1:
            loop_badge = f'<div class="loop-badge">round {revision_round}</div>'

        html += f'''
        <div class="stage">
            <div class="stage-dot {dot_class}">{dot_content}</div>
            <div class="stage-label">{name}</div>
            <div class="stage-sub">{sub}</div>
            {loop_badge}
        </div>'''
        if i < len(AGENTS) - 1:
            line_class = "done" if name in completed_agents else ""
            html += f'<div class="stage-line {line_class}"></div>'
    html += '</div>'
    return html

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
brief = st.text_area(
    "Campaign brief",
    placeholder="e.g. Plan a launch campaign for a budget fitness tracking app targeting college students.",
    height=90,
)
run_button = st.button("Run agent team", type="primary")

# ---------------------------------------------------------------------------
# Run + live pipeline
# ---------------------------------------------------------------------------
if "run" not in st.session_state:
    st.session_state.run = None  # holds {app, config, final_state, awaiting_approval}

if run_button and brief.strip():
    graph_app = build_graph(require_approval=require_approval)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}} if require_approval else None
    initial_state: CampaignState = {
        "brief": brief,
        "max_revisions": max_revisions,
        "revision_count": 0,
        "critique_history": [],
        "agent_log": [],
    }

    pipeline_slot = st.empty()
    log_slot = st.empty()
    final_state = None

    pipeline_slot.markdown(render_pipeline([], None, 1), unsafe_allow_html=True)

    stream_kwargs = {"stream_mode": "values"}
    if config:
        stream_kwargs["config"] = config

    with st.spinner("Agents working..."):
        for step in graph_app.stream(initial_state, **stream_kwargs):
            final_state = step
            log = step.get("agent_log", [])
            completed = [e["agent"] for e in log]
            active = completed[-1] if completed else None
            revision_round = step.get("revision_count", 1) or 1

            pipeline_slot.markdown(
                render_pipeline(completed, active, revision_round),
                unsafe_allow_html=True,
            )

            log_html = ""
            for entry in log:
                icon = next((a[1] for a in AGENTS if a[0] == entry["agent"]), "🤖")
                log_html += f'''
                <div class="log-card">
                    <div class="log-agent">{icon} {entry["agent"]}</div>
                    <div class="log-msg">{md(entry["message"])}</div>
                </div>'''
            log_slot.markdown(log_html, unsafe_allow_html=True)

    st.session_state.run = {
        "app": graph_app,
        "config": config,
        "final_state": final_state,
        # If we required approval and the stream ended without a final_output,
        # the graph is paused at the interrupt right before the Editor node.
        "awaiting_approval": bool(config) and not (final_state or {}).get("final_output"),
    }

run = st.session_state.run

# ---------------------------------------------------------------------------
# Human-in-the-loop approval gate (only shown when paused pre-Editor)
# ---------------------------------------------------------------------------
if run and run["awaiting_approval"]:
    fs = run["final_state"]
    st.markdown("---")
    st.warning(
        "⏸ **Paused for human approval** — the Critic approved this draft "
        f"(score **{fs.get('critique_score', 0):.1f}/10**). Review it below, "
        "then approve to run the Editor's final polish, or reject to ship "
        "the draft as-is."
    )
    with st.expander("Draft awaiting approval", expanded=True):
        st.markdown(md(fs.get("draft", "")))

    col1, col2 = st.columns(2)
    if col1.button("✅ Approve — run Editor polish", type="primary"):
        with st.spinner("Editor polishing..."):
            resumed = run["app"].invoke(None, config=run["config"])
        st.session_state.run["final_state"] = resumed
        st.session_state.run["awaiting_approval"] = False
        st.rerun()
    if col2.button("🚫 Reject — ship draft without Editor polish"):
        fs["final_output"] = fs.get("draft", "")
        st.session_state.run["final_state"] = fs
        st.session_state.run["awaiting_approval"] = False
        st.rerun()

final_state = run["final_state"] if run else None

if run and not run["awaiting_approval"]:
    if final_state and final_state.get("final_output"):
        # mark pipeline fully complete (fresh element - the live `pipeline_slot`
        # placeholder only exists during the run_button branch above; on a
        # rerun triggered by the approve/reject buttons it's gone)
        st.markdown(
            render_pipeline([a[0] for a in AGENTS], None, final_state.get("revision_count", 1)),
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Metric row
        score = final_state.get("critique_score", 0)
        rounds = final_state.get("revision_count", 1)
        n_sources = len(final_state.get("sources", []))
        st.markdown(f'''
        <div class="metric-row">
            <div class="metric-card"><div class="metric-value">{score:.1f}/10</div><div class="metric-label">Final critic score</div></div>
            <div class="metric-card"><div class="metric-value">{rounds}</div><div class="metric-label">Writer revisions</div></div>
            <div class="metric-card"><div class="metric-value">{n_sources}</div><div class="metric-label">Sources used</div></div>
        </div>
        ''', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Campaign Plan", "🔍 Research & Sources", "💰 Budget Fit", "🧐 Critique History",
        ])

        with tab1:
            st.markdown(md(final_state["final_output"]))
            st.download_button(
                "Download as Markdown",
                final_state["final_output"],
                file_name="campaign_plan.md",
                mime="text/markdown",
            )

        with tab2:
            st.markdown(md(final_state.get("research_notes", "")))
            st.markdown("**Sources**")
            for src in final_state.get("sources", []):
                st.markdown(f"- {src}")

        with tab3:
            st.markdown(md(final_state.get("budget_notes", "No budget notes produced.")))

        with tab4:
            history = final_state.get("critique_history", [])
            if not history:
                st.info("No revisions needed — first draft passed the rubric.")
            for i, c in enumerate(history, 1):
                st.markdown(f'<div class="log-card"><div class="log-agent">Round {i}</div><div class="log-msg">{md(c)}</div></div>', unsafe_allow_html=True)
    else:
        st.error("Something went wrong — no final output produced.")

elif run_button:
    st.warning("Please enter a campaign brief first.")
