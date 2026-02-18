from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable

from .types import RewardParts, Task


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class RewardModel:
    """
    Computes a semantic reward with three terms:
    - task_score: external correctness/success signal
    - semantic_score: heuristic structure/reasoning quality
    - format_score: output format adherence
    """

    task_evaluator: Callable[[Task, str], float]
    task_weight: float = 0.7
    semantic_weight: float = 0.2
    format_weight: float = 0.1

    def score(self, task: Task, response: str) -> RewardParts:
        task_score = _clamp(self.task_evaluator(task, response))
        semantic_score = self._semantic_score(response)
        format_score = self._format_score(response)
        total = (
            self.task_weight * task_score
            + self.semantic_weight * semantic_score
            + self.format_weight * format_score
        )
        return RewardParts(
            task_score=task_score,
            semantic_score=semantic_score,
            format_score=format_score,
            total=total,
        )

    @staticmethod
    def _semantic_score(response: str) -> float:
        text = response.lower()
        markers = ["because", "therefore", "step", "verify", "check", "reason"]
        hits = sum(1 for m in markers if m in text)
        return _clamp(hits / 3.0)

    @staticmethod
    def _format_score(response: str) -> float:
        text = response.strip().lower()
        if "answer:" in text:
            return 1.0
        if text:
            return 0.5
        return 0.0


def group_relative_advantage(scores: list[float]) -> list[float]:
    """
    Returns normalized within-group advantages (z-score style).
    If variance is zero, returns zeros.
    """

    if not scores:
        return []
    avg = mean(scores)
    std = pstdev(scores)
    if std == 0.0:
        return [0.0 for _ in scores]
    return [(s - avg) / std for s in scores]
