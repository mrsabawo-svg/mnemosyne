"""
Evaluation Engine: the one job of this module is measuring the thesis.
Every run -- Mnemosyne or baseline -- writes one EvalRecord. This is what
turns "prove the thesis" from a slogan into a comparison you can query.

success/failure alone only measures "did the call complete without
erroring" -- not output quality. quality_score adds a manual 1-5 rating you
can attach after reading a run's outcome, via `python -m mnemosyne.rate`.
"""
from __future__ import annotations

from google.cloud import firestore
from google.cloud.firestore import FieldFilter

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


def rate_run(
    db: firestore.Client, run_id: str, score: float, comment: str = ""
) -> None:
    """Attach a manual quality rating (1-5) to an existing run."""
    if not (1 <= score <= 5):
        raise ValueError("score must be between 1 and 5")
    doc_ref = db.collection("evaluation").document(run_id)
    if not doc_ref.get().exists:
        raise ValueError(f"no evaluation record found for run_id={run_id}")
    doc_ref.update({"quality_score": score, "quality_comment": comment})


def list_recent(db: firestore.Client, limit: int = 10) -> list[dict]:
    """Recent runs, newest first -- useful for finding a run_id to rate."""
    docs = (
        db.collection("evaluation")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [d.to_dict() for d in docs]


def summarize(db: firestore.Client, task_filter: str | None = None) -> dict:
    """Mnemosyne vs baseline, averaged. task_filter does a substring match
    on the task field so trivial smoke-test runs don't dilute the real
    comparison -- e.g. task_filter="ExecutionAgent" isolates just that
    experiment.
    """
    docs = [d.to_dict() for d in db.collection("evaluation").stream()]
    if task_filter:
        docs = [d for d in docs if task_filter.lower() in d.get("task", "").lower()]

    out = {}
    for system in ("mnemosyne", "baseline"):
        rows = [d for d in docs if d.get("system") == system]
        if not rows:
            out[system] = None
            continue

        rated = [r for r in rows if r.get("quality_score") is not None]

        out[system] = {
            "runs": len(rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "avg_retries": sum(r["retries"] for r in rows) / len(rows),
            "avg_interventions": sum(r["user_interventions"] for r in rows)
            / len(rows),
            "avg_duration_seconds": sum(r["duration_seconds"] for r in rows)
            / len(rows),
            "rated_runs": len(rated),
            "avg_quality_score": (
                sum(r["quality_score"] for r in rated) / len(rated)
                if rated
                else None
            ),
        }
    return out
