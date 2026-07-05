"""
Prints the current mnemosyne-vs-baseline comparison. Run this after a batch
of runs on both systems to check progress against the success criteria in
the design doc (thesis validation target: >20% improvement).

Usage:
    python -m mnemosyne.report
    python -m mnemosyne.report --task-filter "ExecutionAgent"
"""
from __future__ import annotations
import argparse
import json

from . import evaluation
from .memory import get_client


def cli() -> None:
    parser = argparse.ArgumentParser(description="Compare mnemosyne vs baseline")
    parser.add_argument(
        "--task-filter",
        default=None,
        help="Substring match on task text -- isolates one experiment from "
        "smoke-test noise, e.g. --task-filter ExecutionAgent",
    )
    args = parser.parse_args()

    db = get_client()
    summary = evaluation.summarize(db, task_filter=args.task_filter)
    print(json.dumps(summary, indent=2))

    m = summary.get("mnemosyne")
    b = summary.get("baseline")
    if m and b and b["success_rate"] > 0:
        improvement = (m["success_rate"] - b["success_rate"]) / b["success_rate"] * 100
        print(f"\nSuccess rate improvement: {improvement:.1f}% (target: >20%)")
    elif m or b:
        print("\nNeed runs from both systems to compare.")

    if m and m.get("avg_quality_score") is not None:
        print(f"Mnemosyne avg quality score: {m['avg_quality_score']:.1f}/5 "
              f"({m['rated_runs']}/{m['runs']} runs rated)")
    if b and b.get("avg_quality_score") is not None:
        print(f"Baseline avg quality score: {b['avg_quality_score']:.1f}/5 "
              f"({b['rated_runs']}/{b['runs']} runs rated)")
    if (not m or m.get("avg_quality_score") is None) and (
        not b or b.get("avg_quality_score") is None
    ):
        print("\nNo quality ratings yet -- success rate above only measures "
              "'completed without erroring', not output quality. Use "
              "`python -m mnemosyne.rate` to add ratings.")


if __name__ == "__main__":
    cli()
