from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass

from .memory import ExperienceMemory
from .openai_backend import OpenAIResponsesBackend
from .reward import RewardModel
from .trainer import TrainerConfig, TrainingFreeGRPOTrainer
from .types import Task

ADD_RE = re.compile(r"(-?\d+)\s*\+\s*(-?\d+)")


@dataclass
class ToyMathBackend:
    """
    Frozen backend with behavior that improves when helpful context exists.
    This simulates the "training-free" setting where only context changes.
    """

    seed: int = 7

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def generate(self, prompt: str, context: str | None = None, temperature: float = 0.7) -> str:
        match = ADD_RE.search(prompt)
        if not match:
            return "I cannot parse this task.\nAnswer: unknown"

        a = int(match.group(1))
        b = int(match.group(2))
        truth = a + b

        context_text = (context or "").lower()
        boosted = "strategy" in context_text or "verify" in context_text
        p_correct = 0.85 if boosted else 0.35

        if self.rng.random() < p_correct:
            answer = truth
            text = (
                "Step 1: add carefully.\n"
                "Step 2: verify arithmetic before finalizing.\n"
                f"Answer: {answer}"
            )
            return text

        answer = truth + self.rng.choice([-2, -1, 1, 2])
        return f"I think it is {answer}.\nAnswer: {answer}"


def math_evaluator(task: Task, response: str) -> float:
    m_task = ADD_RE.search(task.prompt)
    if not m_task:
        return 0.0
    gold = int(m_task.group(1)) + int(m_task.group(2))

    m_ans = re.search(r"answer:\s*(-?\d+)", response.lower())
    if not m_ans:
        return 0.0
    pred = int(m_ans.group(1))
    return 1.0 if pred == gold else 0.0


def run_demo() -> None:
    tasks = [
        Task(task_id="t1", prompt="What is 17 + 26?"),
        Task(task_id="t2", prompt="What is 102 + 9?"),
        Task(task_id="t3", prompt="What is 45 + 44?"),
    ]
    backend = ToyMathBackend(seed=12)
    trainer = TrainingFreeGRPOTrainer(
        backend=backend,
        reward_model=RewardModel(task_evaluator=math_evaluator),
        memory=ExperienceMemory(max_items=100),
        config=TrainerConfig(epochs=4, group_size=6, top_fraction=0.5, min_advantage=0.0),
    )

    report = trainer.train(tasks)
    print("Training report:")
    for e in report.epochs:
        print(
            f"  epoch={e.epoch} rollouts={e.total_rollouts} "
            f"added={e.experiences_added} mean_reward={e.mean_reward:.3f}"
        )

    query = "What is 88 + 15?"
    out = trainer.infer(query)
    print("\nInference with learned token prior")
    print(f"Prompt: {query}")
    print(out)


def run_openai_demo(model: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the OpenAI demo.")

    tasks = [
        Task(task_id="oa1", prompt="What is 37 + 45?"),
        Task(task_id="oa2", prompt="What is 128 + 77?"),
    ]
    backend = OpenAIResponsesBackend(model=model)
    trainer = TrainingFreeGRPOTrainer(
        backend=backend,
        reward_model=RewardModel(task_evaluator=math_evaluator),
        memory=ExperienceMemory(max_items=100),
        config=TrainerConfig(epochs=2, group_size=3, top_fraction=0.5, min_advantage=0.0),
    )

    report = trainer.train(tasks)
    print("OpenAI training report:")
    for e in report.epochs:
        print(
            f"  epoch={e.epoch} rollouts={e.total_rollouts} "
            f"added={e.experiences_added} mean_reward={e.mean_reward:.3f}"
        )

    query = "What is 91 + 16?"
    out = trainer.infer(query)
    print("\nOpenAI inference with learned token prior")
    print(f"Prompt: {query}")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-Free GRPO demo")
    parser.add_argument(
        "--backend",
        choices=["toy", "openai"],
        default="toy",
        help="Backend to use for demo.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Model used when --backend openai.",
    )
    args = parser.parse_args()

    if args.backend == "openai":
        run_openai_demo(model=args.model)
    else:
        run_demo()
