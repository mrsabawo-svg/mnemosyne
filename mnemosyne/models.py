"""
Mnemosyne v1 - core data models.

These mirror the schemas in the design doc but stay minimal for Phase 0.
Every persisted object carries schema_version so future migrations don't
have to guess.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import uuid

SCHEMA_VERSION = 1
AGENT_VERSION = "0.1.0"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class PlanStep:
    order: int
    action: str            # e.g. "search_memory", "reason", "respond"
    description: str
    estimated_effort: int = 1
    status: str = "pending"  # pending | done | failed
    result: Optional[str] = None


@dataclass
class Plan:
    plan_id: str
    goal_id: Optional[str]
    steps: list[PlanStep]
    contingency: str = ""
    created: str = field(default_factory=now_iso)
    schema_version: int = SCHEMA_VERSION


@dataclass
class Episode:
    episode_id: str
    goal_id: Optional[str]
    task: str
    plan_id: Optional[str]
    outcome: str            # short summary of what happened
    success: bool
    steps_taken: int
    retries: int
    user_interventions: int
    timestamp: str = field(default_factory=now_iso)
    embedding: Optional[list[float]] = None
    schema_version: int = SCHEMA_VERSION


@dataclass
class Reflection:
    reflection_id: str
    pattern: str
    gap: str
    next_goal: str
    confidence: float
    affect_delta: dict
    based_on_episodes: list[str]
    timestamp: str = field(default_factory=now_iso)
    schema_version: int = SCHEMA_VERSION


@dataclass
class Goal:
    goal_id: str
    description: str
    priority: float = 0.5
    status: str = "active"   # active | blocked | completed | archived
    parent: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    progress: float = 0.0
    memory_refs: list[str] = field(default_factory=list)
    created: str = field(default_factory=now_iso)
    schema_version: int = SCHEMA_VERSION
    agent_version: str = AGENT_VERSION


@dataclass
class EvalRecord:
    """One row for the Evaluation Engine. This is what proves the thesis."""
    run_id: str
    task: str
    system: str              # "mnemosyne" | "baseline"
    success: bool
    steps_taken: int
    retries: int
    user_interventions: int
    duration_seconds: float
    notes: str = ""
    quality_score: Optional[float] = None  # 1-5, set later via mnemosyne.rate
    quality_comment: str = ""
    timestamp: str = field(default_factory=now_iso)
    schema_version: int = SCHEMA_VERSION



@dataclass
class Affect:
    confidence: float = 0.6
    curiosity: float = 0.5
    urgency: float = 0.3
    confusion: float = 0.2

    def update(self, signal: str, value: float, rate: float = 0.1) -> None:
        old = getattr(self, signal)
        setattr(self, signal, old + rate * (value - old))

    def as_prompt_fragment(self) -> str:
        return (
            f"[affect] confidence={self.confidence:.2f} "
            f"curiosity={self.curiosity:.2f} urgency={self.urgency:.2f} "
            f"confusion={self.confusion:.2f}"
        )


def to_dict(obj) -> dict:
    return asdict(obj)
