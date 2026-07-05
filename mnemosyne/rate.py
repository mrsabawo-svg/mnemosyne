"""
Attach a manual 1-5 quality rating to a run, after reading its outcome.
This is the missing piece that turns "success" from just "didn't error"
into something that also reflects whether the output was actually good.

Usage:
    python -m mnemosyne.rate --list                # see recent run_ids
    python -m mnemosyne.rate <run_id> <score> "optional comment"
"""
from __future__ import annotations
import argparse
import json
import sys

from . import evaluation
from .memory import get_client


def cli() -> None:
    parser = argparse.ArgumentParser(description="Rate a run's output quality")
    parser.add_argument("run_id", nargs="?", help="Run id to rate")
    parser.add_argument("score", nargs="?", type=float, help="Quality score, 1-5")
    parser.add_argument("comment", nargs="?", default="", help="Optional comment")
    parser.add_argument("--list", action="store_true", help="List recent runs")
    args = parser.parse_args()

    db = get_client()

    if args.list or not args.run_id:
        rows = evaluation.list_recent(db, limit=15)
        for r in rows:
            print(
                f"{r['run_id']} | {r['system']:9s} | "
                f"success={r['success']} | "
                f"quality={r.get('quality_score', '-')} | "
                f"task={r['task'][:60]}"
            )
        return

    try:
        evaluation.rate_run(db, args.run_id, args.score, args.comment)
        print(f"Rated {args.run_id}: {args.score}/5 ({args.comment or 'no comment'})")
    except ValueError as e:
        print(f"[rate] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
