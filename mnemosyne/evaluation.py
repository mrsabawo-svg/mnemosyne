"""
Evaluation Engine: the one job of this module is measuring the thesis.
Every run -- Mnemosyne or baseline -- writes one EvalRecord. This is what
turns "prove the thesis" from a slogan into a comparison you can query.
"""
from __future__ import annotations

from google.cloud import firestore

from .models import EvalRecord, new_id, to_dict


def log_run(
    db: firestore.Client,
    task: str,
    system: str,
    success: bool,
    steps_taken: int,
    retries: int,
    user_interventions: int,
    duration_seconds: float,
    notes: str = "",
) -> EvalRecord:
    record = EvalRecord(
        run_id=new_id("run"),
        task=task,
        system=system,
        success=success,
        steps_taken=steps_taken,
        retries=retries,
        user_interventions=user_interventions,
        duration_seconds=duration_seconds,
        notes=notes,
    )
    db.collection("evaluation").document(record.run_id).set(to_dict(record))
    return record


def summarize(db: firestore.Client, task_filter: str | None = None) -> dict:
    """Quick comparison: mnemosyne vs baseline, averaged."""
    query = db.collection("evaluation")
    if task_filter:
        query = query.where("task", "==", task_filter)
    docs = [d.to_dict() for d in query.stream()]

    out = {}
    for system in ("mnemosyne", "baseline"):
        rows = [d for d in docs if d.get("system") == system]
        if not rows:
            out[system] = None
            continue
        out[system] = {
            "runs": len(rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "avg_retries": sum(r["retries"] for r in rows) / len(rows),
            "avg_interventions": sum(r["user_interventions"] for r in rows)
            / len(rows),
            "avg_duration_seconds": sum(r["duration_seconds"] for r in rows)
            / len(rows),
        }
    return out
