from __future__ import annotations

import re
from dataclasses import dataclass, field

from .types import Experience

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union if union else 0.0


@dataclass(slots=True)
class ExperienceMemory:
    """Stores and retrieves distilled experiences used as token priors."""

    max_items: int = 500
    _items: list[Experience] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[Experience]:
        return list(self._items)

    def add(self, experience: Experience) -> None:
        self._items.append(experience)
        self._items.sort(key=lambda e: e.confidence, reverse=True)
        if len(self._items) > self.max_items:
            self._items = self._items[: self.max_items]

    def add_many(self, experiences: list[Experience]) -> None:
        for exp in experiences:
            self.add(exp)

    def retrieve(self, prompt: str, k: int = 5) -> list[Experience]:
        query = _tokens(prompt)
        scored: list[tuple[float, Experience]] = []
        for exp in self._items:
            exp_text = f"{exp.pattern} {' '.join(exp.tags)} {exp.strategy}"
            score = _jaccard(query, _tokens(exp_text)) * max(exp.confidence, 0.01)
            scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for score, exp in scored[:k] if score > 0.0]

    def build_context(self, prompt: str, k: int = 5) -> str:
        retrieved = self.retrieve(prompt, k=k)
        if not retrieved:
            return ""

        lines = ["Learned experiences (token prior):"]
        for idx, exp in enumerate(retrieved, start=1):
            lines.append(f"{idx}. Pattern: {exp.pattern}")
            lines.append(f"   Strategy: {exp.strategy}")
        return "\n".join(lines)
