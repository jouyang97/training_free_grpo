from training_free_grpo.memory import ExperienceMemory
from training_free_grpo.reward import RewardModel, group_relative_advantage
from training_free_grpo.trainer import TrainerConfig, TrainingFreeGRPOTrainer
from training_free_grpo.types import Task


class DummyBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, context: str | None = None, temperature: float = 0.7) -> str:
        self.calls += 1
        if context:
            return "Step 1: reason.\nStep 2: verify.\nAnswer: 10"
        return "Answer: 9"


def _task_evaluator(task: Task, response: str) -> float:
    return 1.0 if "answer: 10" in response.lower() else 0.0


def test_group_relative_advantage_shape() -> None:
    scores = [1.0, 2.0, 3.0]
    adv = group_relative_advantage(scores)
    assert len(adv) == 3
    assert adv[2] > adv[1] > adv[0]


def test_training_adds_experiences_and_infers_with_context() -> None:
    backend = DummyBackend()
    trainer = TrainingFreeGRPOTrainer(
        backend=backend,
        reward_model=RewardModel(task_evaluator=_task_evaluator),
        memory=ExperienceMemory(max_items=20),
        config=TrainerConfig(epochs=2, group_size=4, top_fraction=0.5),
    )

    report = trainer.train([Task(task_id="x", prompt="Compute 5+5")])
    assert report.epochs
    assert len(trainer.memory) > 0

    out = trainer.infer("Compute 6+4")
    assert "Answer:" in out
