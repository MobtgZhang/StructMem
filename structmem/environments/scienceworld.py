from __future__ import annotations

from typing import Any

from structmem.environments.base import BenchmarkEnvironment, InfrastructureError, ResetResult, StepResult
from structmem.environments.compat import bounded_py4j_launch_gateway, patch_scienceworld_partial_close
from structmem.types import TaskSpec


class ScienceWorldEnvironment(BenchmarkEnvironment):
    def __init__(self, split: str, options: dict[str, Any], seed: int):
        self.split = split
        self.options = options
        self.seed = seed
        try:
            from scienceworld import ScienceWorldEnv
            import scienceworld.scienceworld as scienceworld_module
            patch_scienceworld_partial_close(scienceworld_module)
        except Exception as exc:
            raise InfrastructureError(
                "The standard scienceworld package is not importable. Install it normally; path injection is unsupported."
            ) from exc
        try:
            timeout = float(options.get("startup_timeout_seconds", 30))
            scienceworld_module.launch_gateway = lambda *args, **kwargs: bounded_py4j_launch_gateway(
                *args, **kwargs, startup_timeout=timeout
            )
            self.native = ScienceWorldEnv(envStepLimit=int(options.get("engine_max_steps", 100)))
        except Exception as exc:
            raise InfrastructureError(f"ScienceWorld JVM failed to start: {exc}") from exc
        self._last_score = 0.0

    def tasks(self, limit: int) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        for task_name in sorted(self.native.get_task_names()):
            self.native.load(task_name, 0, "")
            variations = {
                "train": self.native.get_variations_train,
                "dev": self.native.get_variations_dev,
                "test": self.native.get_variations_test,
            }[self.split]()
            for variation in variations:
                tasks.append(
                    TaskSpec(
                        dataset="scienceworld",
                        task_id=f"{task_name}:{variation}",
                        goal=task_name,
                        split=self.split,
                        native_id={"task_name": task_name, "variation": int(variation)},
                    )
                )
        selected_ids = self.options.get("task_ids")
        if selected_ids is not None:
            if not isinstance(selected_ids, list) or any(not isinstance(item, str) for item in selected_ids):
                raise InfrastructureError("ScienceWorld options.task_ids must be a list of task-id strings")
            if len(selected_ids) != limit:
                raise InfrastructureError(
                    f"ScienceWorld options.task_ids has {len(selected_ids)} entries, expected tasks={limit}"
                )
            by_id = {task.task_id: task for task in tasks}
            missing = [task_id for task_id in selected_ids if task_id not in by_id]
            if missing:
                raise InfrastructureError(f"Unknown ScienceWorld task ids: {missing}")
            return [by_id[task_id] for task_id in selected_ids]
        if len(tasks) < limit:
            raise InfrastructureError(f"ScienceWorld split {self.split} has {len(tasks)} tasks, fewer than requested {limit}")
        return tasks[:limit]

    def reset(self, task: TaskSpec) -> ResetResult:
        native_id = task.native_id
        try:
            self.native.load(native_id["task_name"], native_id["variation"], "")
            observation, info = self.native.reset()
        except Exception as exc:
            raise InfrastructureError(f"ScienceWorld reset failed for {task.task_id}: {type(exc).__name__}: {exc}") from exc
        self._last_score = float(info.get("score", 0.0))
        return ResetResult(
            observation=observation,
            info={**info, "goal": info.get("taskDesc", task.goal)},
            admissible_actions=None,
        )

    def step(self, action: str) -> StepResult:
        try:
            observation, reward, done, info = self.native.step(action)
        except Exception as exc:
            raise InfrastructureError(f"ScienceWorld engine step failed: {type(exc).__name__}: {exc}") from exc
        score = float(info.get("score", self._last_score + reward))
        self._last_score = score
        success = bool(done and score >= 100.0)
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(done),
            success=success,
            score=score,
            info=info,
            admissible_actions=None,
        )

    def close(self) -> None:
        close = getattr(self.native, "close", None)
        if close:
            close()
