"""
Reflect: after a run, evaluate outcome vs plan/goal and produce a structured
reflection. Fires every run in Phase 0 (the "after 3 interactions OR
confusion > 0.7" trigger logic gets reintroduced once there's enough run
volume for it to matter -- reflecting every run is fine and cheap early on).
"""
from __future__ import annotations

from .llm import generate_json
from .models import Affect, Reflection, new_id

REFLECT_PROMPT = """You are the Reflection module of a task-execution agent.

Task: {task}
Outcome: {outcome}
Success: {success}

Recent episodes for pattern context:
{episodes}

Produce a structured reflection. Respond with ONLY valid JSON:
{{
  "pattern": "one sentence describing any recurring pattern you notice",
  "gap": "one sentence: what the agent doesn't know or can't do yet",
  "next_goal": "one sentence: a concrete next goal worth pursuing, or empty string",
  "confidence": 0.0,
  "affect_delta": {{"confidence": 0.0, "curiosity": 0.0, "urgency": 0.0, "confusion": 0.0}}
}}
"""


def reflect(
    task: str, outcome: str, success: bool, episodes: list[dict], affect: Affect
) -> Reflection:
    episodes_text = "\n".join(
        f"- {e.get('task', '?')} -> {e.get('outcome', '?')} (success={e.get('success')})"
        for e in episodes
    ) or "(none)"

    prompt = REFLECT_PROMPT.format(
        task=task, outcome=outcome, success=success, episodes=episodes_text
    )
    data = generate_json(prompt)

    for key, delta in data.get("affect_delta", {}).items():
        if hasattr(affect, key):
            affect.update(key, getattr(affect, key) + delta)

    return Reflection(
        reflection_id=new_id("refl"),
        pattern=data.get("pattern", ""),
        gap=data.get("gap", ""),
        next_goal=data.get("next_goal", ""),
        confidence=float(data.get("confidence", 0.5)),
        affect_delta=data.get("affect_delta", {}),
        based_on_episodes=[e.get("episode_id", "") for e in episodes],
    )
