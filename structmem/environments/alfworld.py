from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from structmem.environments.base import BenchmarkEnvironment, InfrastructureError, ResetResult, StepResult
from structmem.environments.compat import patch_textworld_eval
from structmem.types import TaskSpec


class ALFWorldEnvironment(BenchmarkEnvironment):
    def __init__(self, split: str, options: dict[str, Any], seed: int):
        self.split = split
        self.options = options
        self.seed = seed
        data_root = options.get("data_root") or os.getenv("ALFWORLD_DATA")
        if not data_root:
            raise InfrastructureError("ALFWorld requires options.data_root or ALFWORLD_DATA")
        self.data_root = Path(os.path.expandvars(data_root)).expanduser().resolve()
        required = self.data_root / "json_2.1.1" / self._split_directory()
        if not required.exists():
            raise InfrastructureError(f"ALFWorld split data not found: {required}")
        os.environ["ALFWORLD_DATA"] = str(self.data_root)
        try:
            from alfworld.agents.environment import get_environment
            patch_textworld_eval()
        except Exception as exc:
            raise InfrastructureError(
                "The standard alfworld package is not importable. Install it normally; path injection is unsupported."
            ) from exc
        default_config = Path(__file__).resolve().parents[1] / "benchmark_data" / "alfworld_base.yaml"
        config_path = Path(options.get("config", default_config))
        if not config_path.exists():
            raise InfrastructureError(f"ALFWorld config not found: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["general"]["random_seed"] = seed
        config["general"]["training_method"] = "dagger"
        config["general"]["use_cuda"] = False
        config["dagger"]["training"]["max_nb_steps_per_episode"] = int(options.get("engine_max_steps", 80))
        environment_class = get_environment(config["env"]["type"])
        self.wrapper = environment_class(config, train_eval=self._native_split())
        self.wrapper.game_files.sort()
        offset = int(self.options.get("offset", 0))
        limit = int(self.options.get("task_limit", 0))
        if offset or limit:
            end = offset + limit if limit else None
            self.wrapper.game_files = self.wrapper.game_files[offset:end]
            self.wrapper.num_games = len(self.wrapper.game_files)
        self.native = self.wrapper.init_env(batch_size=1)
        self._task_position = offset - 1
        self._last_score = 0.0

    def _split_directory(self) -> str:
        return {"train": "train", "valid_seen": "valid_seen", "valid_unseen": "valid_unseen"}[self.split]

    def _native_split(self) -> str:
        return {
            "train": "train",
            "valid_seen": "eval_in_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }[self.split]

    def tasks(self, limit: int) -> list[TaskSpec]:
        files = list(self.wrapper.game_files)
        offset = int(self.options.get("offset", 0))
        selected = files[:limit]
        if len(selected) < limit:
            raise InfrastructureError(
                f"ALFWorld split {self.split} has {len(files)} tasks; offset {offset} leaves fewer than requested {limit}"
            )
        return [
            TaskSpec(
                dataset="alfworld",
                task_id=f"{self.split}:{index:05d}",
                goal="provided by environment on reset",
                split=self.split,
                native_id=index,
                metadata={"game_file": file},
            )
            for index, file in enumerate(selected, start=offset)
        ]

    def reset(self, task: TaskSpec) -> ResetResult:
        expected = self._task_position + 1
        if task.native_id != expected:
            raise InfrastructureError(
                f"ALFWorld adapter is sequential: expected native task {expected}, received {task.native_id}"
            )
        try:
            observations, infos = self.native.reset()
        except Exception as exc:
            raise InfrastructureError(f"ALFWorld reset failed for {task.task_id}: {type(exc).__name__}: {exc}") from exc
        self._task_position = expected
        self._last_score = 0.0
        observation = str(observations[0])
        admissible = list(infos.get("admissible_commands", [[]])[0])
        goal = _goal_from_observation(observation)
        return ResetResult(observation=observation, info={"goal": goal, "native": _first_info(infos)}, admissible_actions=admissible)

    def step(self, action: str) -> StepResult:
        try:
            observations, scores, dones, infos = self.native.step([action])
        except Exception as exc:
            raise InfrastructureError(f"ALFWorld engine step failed: {type(exc).__name__}: {exc}") from exc
        info = _first_info(infos)
        score = float(scores[0])
        reward = score - self._last_score
        self._last_score = score
        success = bool(info.get("won", [False])[0] if isinstance(info.get("won"), list) else info.get("won", False))
        admissible = list(infos.get("admissible_commands", [[]])[0])
        return StepResult(
            observation=str(observations[0]),
            reward=reward,
            done=bool(dones[0]),
            success=success,
            score=score,
            info=info,
            admissible_actions=admissible,
        )

    def close(self) -> None:
        close = getattr(self.native, "close", None)
        if close:
            close()

    def advance(self, task: TaskSpec) -> None:
        self.reset(task)


def _goal_from_observation(observation: str) -> str:
    for line in observation.splitlines():
        if line.lower().startswith(("your task is to", "task:")):
            return line.split(":", 1)[-1].strip()
    return observation.splitlines()[0][:500] if observation else "ALFWorld task"


def _first_info(infos: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in infos.items():
        result[key] = value[0] if isinstance(value, list) and value else value
    return result
