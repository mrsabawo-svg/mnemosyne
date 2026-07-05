"""
Memory layer. Firestore only for v1 (per anti-goals: no Pinecone/Neo4j until
Firestore is proven insufficient).

Since GitHub Actions runs are stateless between triggers, "Working Memory"
isn't a live in-process object that persists across runs -- it's just the
N most recent + most relevant episodes and active goals, re-fetched fresh
at the start of every run. That's still faster than nothing, and it's the
honest version of "working memory" given the runtime constraint.
"""
from __future__ import annotations
import math
import os
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore
from google.cloud.firestore import FieldFilter

from .models import Episode, Goal, Reflection, to_dict

WORKING_MEMORY_SIZE = 5
EPISODE_LOOKBACK_FOR_RETRIEVAL = 50  # cap reads per run to control cost/latency
DATABASE_NAME = "mnemosyne"


def get_client() -> firestore.Client:
    # Expects GOOGLE_APPLICATION_CREDENTIALS env var pointing at a service
    # account key file (written from a GitHub Actions secret at runtime).
    return firestore.Client(database=DATABASE_NAME)


class Memory:
    def __init__(self, db: Optional[firestore.Client] = None):
        self.db = db or get_client()

    # ---------- Episodes ----------

    def save_episode(self, episode: Episode) -> None:
        self.db.collection("episodes").document(episode.episode_id).set(
            to_dict(episode)
        )

    def recent_episodes(self, limit: int = WORKING_MEMORY_SIZE) -> list[dict]:
        docs = (
            self.db.collection("episodes")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [d.to_dict() for d in docs]

    def score_episode(
        self,
        episode: dict,
        active_goal_ids: set[str],
        query_embedding: Optional[list[float]] = None,
    ) -> float:
        """Implements the scoring formula from the design doc. Phase 0 skips
        real embeddings (no vector search wired yet) so semantic_similarity
        defaults to a neutral 0.5 unless embeddings are present on both sides.
        """
        semantic_similarity = 0.5
        if query_embedding and episode.get("embedding"):
            semantic_similarity = _cosine_similarity(
                query_embedding, episode["embedding"]
            )

        goal_id = episode.get("goal_id")
        if goal_id and goal_id in active_goal_ids:
            goal_relevance = 1.0
        elif goal_id:
            goal_relevance = 0.5
        else:
            goal_relevance = 0.0

        ts = episode.get("timestamp")
        recency_decay = _recency_decay(ts)

        reflection_weight = 2.0 if episode.get("_is_reflection") else 1.0
        importance_tag = 1.0 if episode.get("importance") else 0.0

        return (
            semantic_similarity * 0.3
            + goal_relevance * 0.25
            + recency_decay * 0.2
            + reflection_weight * 0.15
            + importance_tag * 0.1
        )

    def retrieve_relevant(
        self, active_goal_ids: set[str], top_k: int = WORKING_MEMORY_SIZE
    ) -> list[dict]:
        candidates = self.recent_episodes(limit=EPISODE_LOOKBACK_FOR_RETRIEVAL)
        scored = [
            (self.score_episode(ep, active_goal_ids), ep) for ep in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    # ---------- Reflections ----------

    def save_reflection(self, reflection: Reflection) -> None:
        self.db.collection("reflections").document(reflection.reflection_id).set(
            to_dict(reflection)
        )

    def recent_reflections(self, limit: int = 3) -> list[dict]:
        docs = (
            self.db.collection("reflections")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [d.to_dict() for d in docs]

    # ---------- Goals ----------

    def save_goal(self, goal: Goal) -> None:
        self.db.collection("goals").document(goal.goal_id).set(to_dict(goal))

    def active_goals(self) -> list[dict]:
        docs = (
            self.db.collection("goals")
            .where(filter=FieldFilter("status", "==", "active"))
            .stream()
        )
        return [d.to_dict() for d in docs]

    def get_goal(self, goal_id: str) -> Optional[dict]:
        doc = self.db.collection("goals").document(goal_id).get()
        return doc.to_dict() if doc.exists else None

    # ---------- Events (log-only, per anti-goals) ----------

    def log_event(self, event_type: str, payload: dict) -> None:
        week = datetime.now(timezone.utc).strftime("%Y-W%U")
        self.db.collection("events").document(week).collection("items").add(
            {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
        )


def _recency_decay(timestamp_str: Optional[str]) -> float:
    if not timestamp_str:
        return 0.0
    try:
        ts = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    hours_since = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    return math.exp(-hours_since / 168)  # 7-day half-life


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
