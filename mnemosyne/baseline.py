"""
Baseline: a plain stateless LLM call on the same task, no memory, no
planner, no reflection. This is what Mnemosyne has to beat to validate
the thesis. Logs to the same evaluation collection under system="baseline"
so evaluation.summarize() can compare them directly.

Usage:
    python -m mnemosyne.baseline "task text"
"""
from __future__ import annotations
import argparse
import sys
import time

from . import evaluation
from .llm import generate
from .memory import get_client


def run(task: str) -> str:
    start = time.time()
    db = get_client()

    try:
        result = generate(task)
        success = True
    except Exception as e:  # noqa: BLE001
        result = f"error: {e}"
        success = False

    duration = time.time() - start
    evaluation.log_run(
        db,
        task=task,
        system="baseline",
        success=success,
        steps_taken=1,
        retries=0,
        user_interventions=0,
        duration_seconds=duration,
    )
    print(f"[baseline] success={success} duration={duration:.1f}s")
    print(f"[baseline] outcome: {result}")
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run one stateless baseline pass")
    parser.add_argument("task", help="Task text for this run")
    args = parser.parse_args()

    try:
        run(args.task)
    except Exception as e:  # noqa: BLE001
        print(f"[baseline] FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()

