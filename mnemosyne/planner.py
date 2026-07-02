"""
Planner: decomposes a task into an ordered sequence of steps.

Phase 0 scope: ordered list only. No DAG, no contingency branching logic
executed (contingency is stored as a hint, not acted on yet) -- per the
anti-goal against building DAG planning before it's needed.
"""
from __future__ import annotations

from .llm import generate_json
from .models import Plan, PlanStep, new_id

PLANNER_PROMPT = """You are the Planner module of a task-execution agent.

Task: {task}

Recent relevant context (most recent episodes and reflections):
{context}

Active goal (if any): {goal}

Decompose this task into 2-5 ordered, concrete steps. Each step should be
something a single tool-call or reasoning pass could accomplish. Do not
solve the task -- only plan it.

Respond with ONLY valid JSON, no markdown fences, in this exact shape:
{{
  "steps": [
    {{"order": 1, "action": "reason", "description": "...", "estimated_effort": 1}}
  ],
  "contingency": "one sentence: what to do if a step fails"
}}

Valid values for "action": "search_memory", "web_search", "reason", "synthesize", "respond".
"""


def make_plan(task: str, context: str, goal: dict | None, goal_id: str | None) -> Plan:
    prompt = PLANNER_PROMPT.format(
        task=task,
        context=context or "(none)",
        goal=goal.get("description") if goal else "(none)",
    )
    data = generate_json(prompt)
    steps = [
        PlanStep(
            order=s["order"],
            action=s["action"],
            description=s["description"],
            estimated_effort=s.get("estimated_effort", 1),
        )
        for s in data["steps"]
    ]
    return Plan(
        plan_id=new_id("plan"),
        goal_id=goal_id,
        steps=steps,
        contingency=data.get("contingency", ""),
    )
