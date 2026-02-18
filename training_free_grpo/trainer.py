from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

from .memory import ExperienceMemory
from .reward import RewardModel, group_relative_advantage
from .types import Experience, LLMBackend, Rollout, Task


@dataclass(slots=True)
class TrainerConfig:
    epochs: int = 3
    group_size: int = 6
    retrieval_k: int = 5
    min_advantage: float = 0.0
    top_fraction: float = 0.5
    temperature: float = 0.7


@dataclass(slots=True)
class EpochStats:
    epoch: int
    total_rollouts: int = 0
    experiences_added: int = 0
    mean_reward: float = 0.0
    mean_advantage: float = 0.0


@dataclass(slots=True)
class TrainingReport:
    epochs: list[EpochStats] = field(default_factory=list)


def _default_experience_extractor(task: Task, selected: list[Rollout], seed: int) -> list[Experience]:
    experiences: list[Experience] = []
    for idx, rollout in enumerate(selected):
        strategy = _extract_strategy(rollout.response)
        pattern = " ".join(task.prompt.split())[:140]
        confidence = max(0.01, min(1.0, 0.5 + 0.25 * rollout.advantage))
        exp = Experience(
            experience_id=f"exp-{seed}-{idx}",
            pattern=pattern,
            strategy=strategy,
            confidence=confidence,
            source_task_id=task.task_id,
            tags=_extract_tags(task.prompt),
        )
        experiences.append(exp)
    return experiences


def _extract_strategy(response: str) -> str:
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    if not lines:
        return "Reason step-by-step, then provide a concise final answer."
    first = lines[0]
    if len(first) > 180:
        first = first[:177] + "..."
    return first


def _extract_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    for key in ("math", "search", "tool", "proof", "code", "plan"):
        if key in lowered:
            tags.append(key)
    return tags


class TrainingFreeGRPOTrainer:
    """
    Training-Free GRPO loop:
    - no model parameter updates
    - grouped rollouts
    - group-relative advantage
    - distill high-advantage trajectories into experience memory
    """

    def __init__(
        self,
        backend: LLMBackend,
        reward_model: RewardModel,
        memory: ExperienceMemory | None = None,
        config: TrainerConfig | None = None,
        extractor: Callable[[Task, list[Rollout], int], list[Experience]] | None = None,
    ) -> None:
        self.backend = backend
        self.reward_model = reward_model
        self.memory = memory or ExperienceMemory()
        self.config = config or TrainerConfig()
        self.extractor = extractor or _default_experience_extractor
        self._exp_counter = itertools.count(1)

    def train(self, tasks: list[Task]) -> TrainingReport:
        report = TrainingReport()
        for epoch in range(1, self.config.epochs + 1):
            epoch_rollouts: list[Rollout] = []
            added = 0
            for task in tasks:
                group = self._sample_group(task)
                selected = self._select_high_advantage(group)
                new_exps = self.extractor(task, selected, next(self._exp_counter))
                self.memory.add_many(new_exps)
                added += len(new_exps)
                epoch_rollouts.extend(group)

            mean_reward = (
                sum(r.rewards.total for r in epoch_rollouts) / len(epoch_rollouts)
                if epoch_rollouts
                else 0.0
            )
            mean_advantage = (
                sum(r.advantage for r in epoch_rollouts) / len(epoch_rollouts)
                if epoch_rollouts
                else 0.0
            )
            report.epochs.append(
                EpochStats(
                    epoch=epoch,
                    total_rollouts=len(epoch_rollouts),
                    experiences_added=added,
                    mean_reward=mean_reward,
                    mean_advantage=mean_advantage,
                )
            )
        return report

    def infer(self, prompt: str) -> str:
        context = self.memory.build_context(prompt, k=self.config.retrieval_k)
        return self.backend.generate(prompt=prompt, context=context, temperature=0.0)

    def _sample_group(self, task: Task) -> list[Rollout]:
        group: list[Rollout] = []
        context = self.memory.build_context(task.prompt, k=self.config.retrieval_k)
        for _ in range(self.config.group_size):
            response = self.backend.generate(
                prompt=task.prompt,
                context=context,
                temperature=self.config.temperature,
            )
            rewards = self.reward_model.score(task, response)
            group.append(
                Rollout(
                    task_id=task.task_id,
                    response=response,
                    context=context,
                    rewards=rewards,
                )
            )

        advantages = group_relative_advantage([g.rewards.total for g in group])
        for rollout, adv in zip(group, advantages):
            rollout.advantage = adv
        return group

    def _select_high_advantage(self, group: list[Rollout]) -> list[Rollout]:
        if not group:
            return []
        sorted_group = sorted(group, key=lambda r: r.advantage, reverse=True)
        keep_n = max(1, int(len(group) * self.config.top_fraction))
        selected = sorted_group[:keep_n]
        return [r for r in selected if r.advantage >= self.config.min_advantage]
