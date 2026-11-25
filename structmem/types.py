from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    INFORMATION_ACQUISITION = "InformationAcquisition"
    PRECONDITION_CHECK = "PreconditionCheck"
    STATE_CHANGING_ACTION = "StateChangingAction"
    VERIFICATION = "Verification"
    RECOVERY = "Recovery"
    TERMINATION = "Termination"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    TASK_FAILED = "task_failed"
    AGENT_ERROR = "agent_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class Cell(str, Enum):
    NEAR_NEAR = "near_semantic_near_structural"
    FAR_NEAR = "far_semantic_near_structural"
    NEAR_FAR = "near_semantic_far_structural"
    FAR_FAR = "far_semantic_far_structural"


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task_id: str
    goal: str
    split: str
    native_id: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    observation: str
    action: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    task: TaskSpec
    method: str
    status: RunStatus
    success: bool
    score: float
    steps: list[Step]
    error: str | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    transfer_used: bool = False
    binding_outcome: str | None = None
    gate_verdict: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class Experience:
    experience_id: str
    dataset: str
    task_id: str
    goal: str
    success: bool
    score: float
    actions: list[str]
    observations: list[str]
    roles: list[Role]
    graph: dict[str, Any]
    lesson: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["roles"] = [role.value for role in self.roles]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        copy = dict(data)
        copy["roles"] = [Role(role) for role in copy.get("roles", [])]
        return cls(**copy)
