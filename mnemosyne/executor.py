"""
THINK + ACT: executes each plan step.

Phase 0 has a small action registry. "web_search" and "search_memory" are
stubbed to fall back to reasoning if not wired to a real tool yet -- this
keeps the loop runnable end-to-end before every tool integration exists,
which is the point of Phase 0 (prove the loop, not the tool coverage).
"""
from __future__ import annotations

from .llm import generate
from .models import Affect, PlanStep

MAX_RETRIES = 3


def execute_step(
    step: PlanStep,
    task: str,
    context: str,
    affect: Affect,
) -> tuple[bool, str, int]:
    """Returns (success, result_text, retries_used)."""
    retries = 0
    last_error = ""
    while retries <= MAX_RETRIES:
        try:
            result = _run_action(step, task, context, affect)
            return True, result, retries
        except Exception as e:  # noqa: BLE001 - deliberately broad for Phase 0
            last_error = str(e)
            retries += 1
    return False, f"failed after {retries} retries: {last_error}", retries


def _run_action(step: PlanStep, task: str, context: str, affect: Affect) -> str:
    if step.action in ("reason", "synthesize", "respond"):
        return _reason(step, task, context, affect)
    if step.action in ("search_memory", "web_search"):
        # Not wired to a real search tool in Phase 0 -- reason about it
        # instead so the loop stays runnable. Replace with a real tool
        # call once web_search / memory retrieval are integrated here.
        return _reason(step, task, context, affect)
    raise ValueError(f"unknown action: {step.action}")


def _reason(step: PlanStep, task: str, context: str, affect: Affect) -> str:
    prompt = f"""{affect.as_prompt_fragment()}

Task: {task}
Current step ({step.action}): {step.description}

Relevant context:
{context or '(none)'}

Carry out this step. Be direct and concrete. If this is the final
"respond" step, produce the actual answer/output for the task.
"""
    return generate(prompt)

