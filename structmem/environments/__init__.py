from __future__ import annotations

from typing import Any

from structmem.environments.agentbench import AgentBenchEnvironment
from structmem.environments.alfworld import ALFWorldEnvironment
from structmem.environments.base import BenchmarkEnvironment
from structmem.environments.scienceworld import ScienceWorldEnvironment


def make_environment(name: str, split: str, options: dict[str, Any], seed: int) -> BenchmarkEnvironment:
    classes = {
        "alfworld": ALFWorldEnvironment,
        "scienceworld": ScienceWorldEnvironment,
        "agentbench": AgentBenchEnvironment,
    }
    try:
        cls = classes[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc
    return cls(split, options, seed)
