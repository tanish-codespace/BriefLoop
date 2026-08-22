"""
Evaluation harness.

Runs the full pipeline over a fixed set of sample briefs and measures the
thing the whole self-correcting-loop design is supposed to buy you: does
the Critic->Writer feedback loop actually improve the draft, or is it just
burning API calls?

For each brief we record:
- first-draft critic score vs final critic score (the delta is the whole
  point of the loop - it should usually be >= 0)
- how many writer revisions it took
- how many sources the researcher pulled in
- wall-clock latency

Usage:
    python -m evals.run_eval
    python -m evals.run_eval --limit 2 --max-revisions 3
    python -m evals.run_eval --briefs evals/sample_briefs.json

Requires real GROQ_API_KEY / TAVILY_API_KEY in .env - this hits live APIs
(it's an eval script, not a unit test; see tests/ for the mocked suite that
runs in CI without keys).
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import run_campaign  # noqa: E402

SCORE_RE = re.compile(r"Score ([\d.]+)/10")


def _first_score(critique_history: list[str]) -> float | None:
    if not critique_history:
        return None
    match = SCORE_RE.search(critique_history[0])
    return float(match.group(1)) if match else None


def run_eval(briefs_path: Path, limit: int | None, max_revisions: int) -> list[dict]:
    briefs = json.loads(briefs_path.read_text(encoding="utf-8"))
    if limit:
        briefs = briefs[:limit]

    results = []
    for item in briefs:
        print(f"Running: {item['id']} ...", end=" ", flush=True)
        start = time.perf_counter()
        try:
            state = run_campaign(item["brief"], max_revisions=max_revisions)
            elapsed = time.perf_counter() - start
            first = _first_score(state.get("critique_history", []))
            final = state.get("critique_score")
            results.append({
                "id": item["id"],
                "brief": item["brief"],
                "first_score": first,
                "final_score": final,
                "score_delta": (final - first) if (first is not None and final is not None) else None,
                "revisions": state.get("revision_count"),
                "num_sources": len(state.get("sources", [])),
                "seconds": round(elapsed, 2),
                "error": None,
            })
            print(f"first={first} -> final={final} ({elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - one bad brief shouldn't kill the run
            elapsed = time.perf_counter() - start
            results.append({
                "id": item["id"], "brief": item["brief"], "first_score": None,
                "final_score": None, "score_delta": None, "revisions": None,
                "num_sources": None, "seconds": round(elapsed, 2), "error": str(exc),
            })
            print(f"FAILED: {exc}")

    return results


def print_summary(results: list[dict]) -> None:
    ok = [r for r in results if r["error"] is None]
    print("\n" + "=" * 72)
    print(f"{'id':<24} {'first':>6} {'final':>6} {'delta':>6} {'revs':>5} {'srcs':>5} {'time':>7}")
    print("-" * 72)
    for r in results:
        if r["error"]:
            print(f"{r['id']:<24} ERROR: {r['error']}")
            continue
        print(
            f"{r['id']:<24} {r['first_score']:>6} {r['final_score']:>6} "
            f"{r['score_delta']:>+6.1f} {r['revisions']:>5} {r['num_sources']:>5} {r['seconds']:>6.1f}s"
        )
    print("-" * 72)
    if ok:
        deltas = [r["score_delta"] for r in ok if r["score_delta"] is not None]
        finals = [r["final_score"] for r in ok if r["final_score"] is not None]
        if deltas:
            print(f"Avg score improvement from revision loop: {statistics.mean(deltas):+.2f}")
        if finals:
            print(f"Avg final score: {statistics.mean(finals):.2f}/10")
    failed = len(results) - len(ok)
    if failed:
        print(f"{failed} brief(s) failed - see errors above.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Run the campaign pipeline over sample briefs and score it.")
    parser.add_argument("--briefs", type=Path, default=Path(__file__).parent / "sample_briefs.json")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N briefs")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    load_dotenv()

    results = run_eval(args.briefs, args.limit, args.max_revisions)
    print_summary(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
