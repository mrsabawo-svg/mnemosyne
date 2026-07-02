"""
Prints the current mnemosyne-vs-baseline comparison. Run this after a batch
of runs on both systems to check progress against the success criteria in
the design doc (thesis validation target: >20% improvement).

Usage:
    python -m mnemosyne.report
"""
from __future__ import annotations
import json

from . import evaluation
from .memory import get_client


def cli() -> None:
    db = get_client()
    summary = evaluation.summarize(db)
    print(json.dumps(summary, indent=2))

    m = summary.get("mnemosyne")
    b = summary.get("baseline")
    if m and b and b["success_rate"] > 0:
        improvement = (m["success_rate"] - b["success_rate"]) / b["success_rate"] * 100
        print(f"\nSuccess rate improvement: {improvement:.1f}% "
              f"(target: >20%)")
    elif m or b:
        print("\nNeed runs from both systems to compare.")


if __name__ == "__main__":
    cli()
