from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class LLMBackend(Protocol):
    """Backend interface for a frozen language model."""

    def generate(self, prompt: str, context: str | None = None, temperature: float = 0.7) -> str:
        """Generate one response from a frozen model."""


@dataclass(slots=True)
class Task:
    task_id: str
    prompt: str
    reference: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RewardParts:
    task_score: float
    semantic_score: float
    format_score: float
    total: float


@dataclass(slots=True)
class Rollout:
    task_id: str
    response: str
    context: str
    rewards: RewardParts
    advantage: float = 0.0


@dataclass(slots=True)
class Experience:
    experience_id: str
    pattern: str
    strategy: str
    confidence: float
    source_task_id: str
    tags: list[str] = field(default_factory=list)
