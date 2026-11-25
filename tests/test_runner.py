from pathlib import Path

from structmem.agents.base import Agent, AgentContext
from structmem.config import APIConfig, DatasetConfig, ExperimentConfig
from structmem.environments.base import ResetResult, StepResult
from structmem import runner as runner_module
from structmem.runner import ExperimentRunner
from structmem.types import Experience, RunStatus, TaskSpec


class FrozenAgent(Agent):
    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        return AgentContext(task=task)


def config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=42,
        output_dir=tmp_path,
        methods=("no_memory",),
        datasets={},
        api=APIConfig("http://example.invalid/v1", "key", "model"),
    )


def test_online_updates_are_off_by_default(tmp_path) -> None:
    runner = ExperimentRunner(config(tmp_path), tmp_path / "config.yaml")
    assert not runner.config.online_memory_updates
    assert not FrozenAgent(runner.llm).allow_online_updates


class CountingAgent(FrozenAgent):
    name = "no_memory"

    def begin(self, task, query, memory):
        context = super().begin(task, query, memory)
        context.model_usage = {"input_tokens": 5, "output_tokens": 2}
        return context

    def act(self, context, observation, admissible_actions):
        return "act", {"input_tokens": 3, "output_tokens": 1}


class FakeEnvironment:
    def __init__(self):
        self.advanced = []
        self.reset_ids = []

    def tasks(self, limit):
        return [TaskSpec("fake", f"t{index}", f"goal {index}", "test", index) for index in range(limit)]

    def advance(self, task):
        self.advanced.append(task.task_id)

    def reset(self, task):
        self.reset_ids.append(task.task_id)
        return ResetResult("observation", {"goal": task.goal})

    def step(self, action):
        return StepResult("done", 1.0, True, True, 1.0)

    def close(self):
        pass


def test_episode_aggregates_internal_and_action_model_usage(tmp_path) -> None:
    runner = ExperimentRunner(config(tmp_path), tmp_path / "config.yaml")
    task = TaskSpec("fake", "t", "goal", "test", 0)
    episode = runner._episode(FakeEnvironment(), CountingAgent(runner.llm), task, 1)
    assert episode.status == RunStatus.COMPLETED
    assert episode.token_usage == {"input_tokens": 8, "output_tokens": 3}


def test_resume_advances_stateful_environment(tmp_path, monkeypatch) -> None:
    cfg = ExperimentConfig(
        seed=42,
        output_dir=tmp_path,
        methods=("no_memory",),
        datasets={"fake": DatasetConfig(True, "test", tasks=2, max_steps=1)},
        api=APIConfig("http://example.invalid/v1", "key", "model"),
    )
    runner = ExperimentRunner(cfg, tmp_path / "config.yaml")
    runner.store.append(
        {
            "task": {"dataset": "fake", "task_id": "t0"},
            "method": "no_memory",
            "status": "completed",
        }
    )
    environment = FakeEnvironment()
    monkeypatch.setattr(runner_module, "make_environment", lambda *args, **kwargs: environment)
    monkeypatch.setattr(
        runner_module,
        "make_agent",
        lambda *args, **kwargs: CountingAgent(runner.llm),
    )
    runner.run()
    assert environment.advanced == ["t0"]
    assert environment.reset_ids == ["t1"]


def test_resume_uses_external_completed_keys_without_copying_results(tmp_path, monkeypatch) -> None:
    completed_keys_file = tmp_path / "completed.json"
    completed_keys_file.write_text(
        '{"keys":[{"dataset":"fake","method":"no_memory","task_id":"t0"}]}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "new-results"
    cfg = ExperimentConfig(
        seed=42,
        output_dir=output_dir,
        methods=("no_memory",),
        datasets={"fake": DatasetConfig(True, "test", tasks=2, max_steps=1)},
        api=APIConfig("http://example.invalid/v1", "key", "model"),
        completed_keys_file=completed_keys_file,
    )
    runner = ExperimentRunner(cfg, tmp_path / "config.yaml")
    environment = FakeEnvironment()
    monkeypatch.setattr(runner_module, "make_environment", lambda *args, **kwargs: environment)
    monkeypatch.setattr(runner_module, "make_agent", lambda *args, **kwargs: CountingAgent(runner.llm))

    runner.run()

    assert environment.advanced == ["t0"]
    assert environment.reset_ids == ["t1"]
    rows = runner.store.read()
    assert [row["task"]["task_id"] for row in rows] == ["t1"]
