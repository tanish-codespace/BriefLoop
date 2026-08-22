# Multi-Agent Marketing Campaign Planner

![CI](https://github.com/USERNAME/marketing-agents/actions/workflows/ci.yml/badge.svg)

A small team of specialized LLM agents that plan a marketing campaign
end-to-end, with a genuine self-correcting feedback loop — not just a
scripted linear chain. Built with **LangGraph**, tested with **pytest**
(23 tests, fully mocked — no API keys needed to run CI), and shippable via
**Docker**.

> Update the badge URL above (`USERNAME/marketing-agents`) once this is
> pushed to your own GitHub repo.

## Architecture

```
        START
          |
     ┌────▼─────┐
     │Researcher│  gathers real web data (Tavily search), synthesizes notes
     └────┬─────┘
          |
   ┌──────▼───────┐
   │Budget Analyst│  turns the brief into a concrete budget constraint +
   └──────┬───────┘  a channel allocation that fits it
          |
     ┌────▼─────┐
     │  Writer   │◄────────────┐  drafts the campaign plan
     └────┬─────┘              │
          |                    │ (if score < 7.5 and revisions left)
     ┌────▼─────┐              │
     │  Critic   │──────────────┘  scores draft 0-10 against a rubric,
     └────┬─────┘                  gives specific actionable feedback
          | (score >= 7.5, or revision cap hit)
   [ optional human approval gate ]
          |
     ┌────▼─────┐
     │  Editor   │  final polish pass, adds executive summary
     └────┬─────┘
          |
         END
```

This is a `StateGraph` with a conditional edge on the Critic node. The key
design decision worth calling out: routing is decided *programmatically*
from the Critic's structured score, not by another LLM call guessing what
to do next — makes the control flow deterministic and debuggable.

## Why this design (things worth mentioning in an interview)

- **Real feedback loop, capped.** The Writer→Critic→Writer loop actually
  changes the draft based on specific critique, not just a rubber-stamp
  "looks good." It's capped by `max_revisions` so it can't spiral — a
  guardrail every agentic system needs. Covered by
  [`tests/test_graph_integration.py`](tests/test_graph_integration.py),
  which asserts the exact node sequence for both the happy path and the
  loop-then-approve path.
- **Structured critic output.** The Critic returns JSON (`score`,
  `feedback`), not prose, so the graph can route on it reliably. There's a
  tested fallback if the model doesn't return clean JSON, so a parsing
  hiccup doesn't crash the whole pipeline.
- **Grounded, not just prompted.** The Researcher calls a real search API
  and the Writer/Critic are explicitly told to check the draft against the
  research notes *and* the budget notes — this cuts down on generic,
  hallucinated campaign advice.
- **Budget-aware, not just channel-happy.** The Budget Analyst agent
  extracts a stated budget from the brief (or transparently infers one),
  and the Writer/Critic are both required to respect it — a plan that
  recommends TV ads on a $500 budget now gets caught and rejected, not
  shipped.
- **Human-in-the-loop is opt-in, not bolted on.** `build_graph(require_approval=True)`
  compiles the graph with a `MemorySaver` checkpointer and an
  `interrupt_before=["editor"]`, so a caller can pause right before the
  final polish pass, show the critic-approved draft to a human, and resume
  from saved state (`app.invoke(None, config)`) rather than re-running the
  whole pipeline. Same graph definition either way — approval is a compile
  option, not a fork of the codebase.
- **Separation of concerns.** Each agent has one job and one prompt. Easy to
  swap models per-agent (e.g. a cheaper/faster model for the Critic) or add
  a new agent without touching the others — the Budget Analyst was added
  this way, as a single new node + one edge, with zero changes to the other
  three agents' internals.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in your API keys
```

You'll need (both free, no credit card required):
- A **Groq API key** (console.groq.com — free tier, runs an open-weight
  120B-parameter model)
- A **Tavily API key** (tavily.com — free tier is enough for this)

## Run it

**Streamlit UI (recommended for demos):**
```bash
streamlit run app.py
```
Check "Require human approval before final polish" in the sidebar to see
the human-in-the-loop gate in action.

**Command line:**
```bash
python graph.py "Plan a launch campaign for a budget fitness tracking app targeting college students."

# with a human approval gate before the Editor runs:
python graph.py --review "Plan a launch campaign for a budget fitness tracking app targeting college students."
```

**Docker:**
```bash
docker build -t marketing-agents .
docker run --env-file .env -p 8501:8501 marketing-agents
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v          # 23 tests, all mocked — no API keys required
ruff check .        # lint
```

Test coverage is organized the way the pipeline is organized: one file per
agent for isolated node tests (prompt selection, JSON-parsing fallback,
count increments), plus [`tests/test_graph_integration.py`](tests/test_graph_integration.py)
and [`tests/test_human_in_the_loop.py`](tests/test_human_in_the_loop.py)
for the actual graph wiring — the happy path, the revise-then-approve loop,
the revision-cap force-through, and the interrupt/resume mechanics. All LLM
and search calls are mocked at the module boundary (`monkeypatch.setattr`
on `get_llm`/`web_search`), so the suite is fast, free, and safe to run in
CI with no secrets. [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs it (plus `ruff`) on every push and PR.

Writing these tests caught two real bugs during development: a regex that
left a trailing space in extracted budget figures, and a test-mock bug
that silently reset a fake LLM's call history between graph steps — a good
reminder that mocked tests need care too.

## Evaluation harness

Unit tests check that the pipeline *works*. They don't tell you whether the
Critic→Writer loop actually makes plans *better*. [`evals/run_eval.py`](evals/run_eval.py)
does that: it runs the full pipeline (real API calls) over a fixed set of
sample briefs in [`evals/sample_briefs.json`](evals/sample_briefs.json) and
reports, per brief, the first-draft critic score vs. the final score, the
number of revisions it took, and wall-clock latency — then saves the raw
results as JSON so runs are comparable over time.

```bash
python -m evals.run_eval                       # all sample briefs
python -m evals.run_eval --limit 2              # quick smoke run
python -m evals.run_eval --max-revisions 3
```

This is the kind of lightweight eval loop that's easy to talk about in an
interview: it turns "the agent got better" from a vibe into a number you
can track across prompt changes.

## Human-in-the-loop

Fully autonomous is the default, but `build_graph(require_approval=True)`
adds a real pause-and-resume gate right before the Editor node using
LangGraph's checkpointer + `interrupt_before`. The CLI's `--review` flag and
the Streamlit sidebar checkbox both use this — same graph, same nodes, no
duplicated pipeline logic for the "supervised" mode.

## Project structure

```
marketing-agents/
├── agents/
│   ├── researcher.py       # web search + synthesis
│   ├── budget_analyst.py   # budget constraint + channel allocation
│   ├── writer.py           # drafts + revises
│   ├── critic.py           # scores + gives feedback
│   └── editor.py           # final polish
├── graph.py                # LangGraph wiring, routing logic, HITL support
├── state.py                # shared state schema (TypedDict)
├── tools.py                # Tavily search wrapper
├── llm.py                  # centralized model config
├── app.py                  # Streamlit front end (incl. approval UI)
├── tests/                  # pytest suite, fully mocked
├── evals/                  # eval harness + sample briefs
├── Dockerfile
├── .github/workflows/ci.yml
└── requirements.txt / requirements-dev.txt
```

## Possible extensions (good "future work" talking points)

- Swap the fixed threshold (7.5) for a **learned or adaptive threshold**
  based on the campaign type.
- Track eval results over time (append to a small SQLite/CSV log instead of
  one JSON file per run) and plot first-vs-final score trends as prompts
  change.
- Add a **second, independent critic model** ("LLM jury") and require
  agreement before auto-approving, as an alternative to the human approval
  gate for higher-stakes campaigns.
- Stream token-by-token output in the Streamlit UI instead of per-node
  snapshots, for a more "live" feel during long research/writing steps.

## How to talk about this on a resume

A couple of framings that map directly to what's actually implemented here
(swap in real numbers from your own eval runs):

- *"Built a multi-agent marketing planning pipeline (LangGraph) with a
  programmatic critic→writer feedback loop, an optional human-in-the-loop
  approval gate via checkpointed interrupts, and a budget-aware planning
  agent; covered by a 23-test mocked pytest suite running in CI and a
  standalone evaluation harness measuring draft-quality improvement across
  revision rounds."*
- *"Designed the control flow so revision routing is decided by parsed,
  structured model output (JSON score) rather than another LLM call, with
  a tested fallback path for malformed responses — a small but concrete
  example of making an agentic system's control flow deterministic and
  debuggable instead of purely LLM-driven."*
