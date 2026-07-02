"""
Orchestrator entry point. One invocation = one full loop pass:

    SENSE -> PLANNER -> THINK/ACT -> REFLECT -> store episode -> update goal

Triggered by GitHub Actions (workflow_dispatch or repository_dispatch via
cronjob.org, same pattern as OBI). State lives entirely in Firestore between
runs since Actions gives no persistent process.

Usage:
    python -m mnemosyne.main "task text" [--goal-id GOAL_ID]
"""
from __future__ import annotations
import argparse
import sys
import time

from . import evaluation, planner, reflect
from .executor import execute_step
from .memory import Memory
from .models import Affect, Episode, new_id


def run(task: str, goal_id: str | None = None) -> Episode:
    start = time.time()
    mem = Memory()

    # ---- SENSE ----
    active_goals = mem.active_goals()
    active_goal_ids = {g["goal_id"] for g in active_goals}
    goal = mem.get_goal(goal_id) if goal_id else None

    relevant_episodes = mem.retrieve_relevant(active_goal_ids)
    recent_reflections = mem.recent_reflections()
    context = _build_context(relevant_episodes, recent_reflections)

    affect = Affect()  # Phase 0: fresh affect per run. Persisting affect
    # across runs via the identity document lands in Phase 4.

    # ---- PLANNER ----
    plan = planner.make_plan(task, context, goal, goal_id)

    # ---- THINK / ACT ----
    total_retries = 0
    step_results = []
    all_succeeded = True
    for step in plan.steps:
        success, result, retries = execute_step(step, task, context, affect)
        step.status = "done" if success else "failed"
        step.result = result
        total_retries += retries
        step_results.append(result)
        if not success:
            all_succeeded = False
            break  # Phase 0: stop on first hard failure, per orchestrator
            # decision table ("ask user" / "defer to goal" land later)

    outcome = step_results[-1] if step_results else "(no steps executed)"

    # ---- REFLECT ----
    episode = Episode(
        episode_id=new_id("ep"),
        goal_id=goal_id,
        task=task,
        plan_id=plan.plan_id,
        outcome=outcome,
        success=all_succeeded,
        steps_taken=len(step_results),
        retries=total_retries,
        user_interventions=0,
    )
    mem.save_episode(episode)
    mem.log_event("EpisodeCompleted", {"episode_id": episode.episode_id})

    reflection = reflect.reflect(
        task, outcome, all_succeeded, relevant_episodes, affect
    )
    mem.save_reflection(reflection)
    mem.log_event("ReflectionGenerated", {"reflection_id": reflection.reflection_id})

    if goal_id and goal:
        _update_goal_progress(mem, goal_id, all_succeeded)

    # ---- Evaluation Engine ----
    duration = time.time() - start
    evaluation.log_run(
        mem.db,
        task=task,
        system="mnemosyne",
        success=all_succeeded,
        steps_taken=len(step_results),
        retries=total_retries,
        user_interventions=0,
        duration_seconds=duration,
        notes=reflection.gap,
    )

    print(f"[mnemosyne] success={all_succeeded} steps={len(step_results)} "
          f"retries={total_retries} duration={duration:.1f}s")
    print(f"[mnemosyne] outcome: {outcome}")
    print(f"[mnemosyne] reflection.gap: {reflection.gap}")

    return episode


def _build_context(episodes: list[dict], reflections: list[dict]) -> str:
    lines = []
    if episodes:
        lines.append("Past episodes:")
        for e in episodes:
            lines.append(f"- {e.get('task')} -> {e.get('outcome')}")
    if reflections:
        lines.append("Recent reflections:")
        for r in reflections:
            if r.get("gap"):
                lines.append(f"- gap: {r['gap']}")
            if r.get("next_goal"):
                lines.append(f"- suggested next goal: {r['next_goal']}")
    return "\n".join(lines)


def _update_goal_progress(mem: Memory, goal_id: str, step_succeeded: bool) -> None:
    goal = mem.get_goal(goal_id)
    if not goal:
        return
    increment = 0.1 if step_succeeded else 0.0
    goal["progress"] = min(1.0, goal.get("progress", 0.0) + increment)
    if goal["progress"] >= 1.0:
        goal["status"] = "completed"
    mem.db.collection("goals").document(goal_id).set(goal)
    mem.log_event("GoalProgressUpdated", {"goal_id": goal_id, "progress": goal["progress"]})


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run one Mnemosyne loop pass")
    parser.add_argument("task", help="Task text for this run")
    parser.add_argument("--goal-id", default=None, help="Optional goal to link this run to")
    args = parser.parse_args()

    try:
        run(args.task, args.goal_id)
    except Exception as e:  # noqa: BLE001
        print(f"[mnemosyne] FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
