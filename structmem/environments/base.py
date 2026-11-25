from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from structmem.types import TaskSpec


class InfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResetResult:
    observation: str
    info: dict[str, Any] = field(default_factory=dict)
    admissible_actions: list[str] | None = None


@dataclass(frozen=True)
class StepResult:
    observation: str
    reward: float
    done: bool
    success: bool
    score: float
    info: dict[str, Any] = field(default_factory=dict)
    admissible_actions: list[str] | None = None


class BenchmarkEnvironment(ABC):
    @abstractmethod
    def tasks(self, limit: int) -> list[TaskSpec]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task: TaskSpec) -> ResetResult:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def advance(self, task: TaskSpec) -> None:
        """Advance stateful sequential environments past a resumed task."""

