from __future__ import annotations

import socket
import json
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from structmem.config import ExperimentConfig
from structmem.environments import make_environment
from structmem.environments.base import InfrastructureError
from structmem.factory import METHODS
from structmem.memory import ExperienceStore


@dataclass(frozen=True)
class Check:
    component: str
    ok: bool
    detail: str
    capacity: int | None = None


def preflight(config: ExperimentConfig) -> dict[str, Any]:
    checks: list[Check] = []
    unknown = sorted(set(config.methods) - set(METHODS))
    checks.append(Check("methods", not unknown, f"configured={len(config.methods)}; unknown={unknown}"))
    host = urlparse(config.api.base_url).hostname
    try:
        if not host:
            raise ValueError("API URL has no hostname")
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, 443)})
        checks.append(Check("llm_dns", True, f"{host} resolved to {', '.join(addresses)}"))
    except Exception as exc:
        checks.append(Check("llm_dns", False, f"{host}: {type(exc).__name__}: {exc}"))
    for name, dataset in config.datasets.items():
        if not dataset.enabled:
            continue
        environment = None
        try:
            options = {**dataset.options, "task_limit": dataset.tasks}
            environment = make_environment(name, dataset.split, options, config.seed)
            tasks = environment.tasks(dataset.tasks)
            checks.append(Check(name, True, f"split={dataset.split}; requested task ids enumerated", len(tasks)))
            pair_value = dataset.options.get("pair_manifest")
            if pair_value:
                pair_path = Path(str(pair_value)).expanduser()
                if not pair_path.is_absolute():
                    checks.append(Check(f"{name}_pairs", False, f"pair path was not resolved: {pair_path}"))
                else:
                    checks.append(_pair_check(name, pair_path, tasks))
        except Exception as exc:
            checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
        finally:
            if environment:
                environment.close()
    if config.memory_file:
        memory = ExperienceStore.load(config.memory_file) if config.memory_file.exists() else ExperienceStore()
        checks.append(
            Check(
                "memory",
                config.memory_file.exists() and bool(memory.values()),
                f"{config.memory_file}; experiences={len(memory.values())}",
            )
        )
    else:
        memory_methods = set(config.methods) - {"no_memory", "reflexion", "expel"}
        checks.append(
            Check(
                "memory",
                not memory_methods,
                "No memory_file configured; retrieval methods will have empty memory",
            )
        )
    requires_dense = any(
        method in {"embedding_retrieval", "ours"} or method.startswith("w_o_")
        for method in config.methods
    )
    if requires_dense:
        try:
            from sentence_transformers import SentenceTransformer

            SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            checks.append(Check("dense_embedding", True, "local all-MiniLM-L6-v2 model loaded"))
        except Exception as exc:
            checks.append(Check("dense_embedding", False, f"{type(exc).__name__}: {exc}"))
    return {
        "ready": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def _pair_check(dataset: str, path: Path, tasks: list[Any]) -> Check:
    if not path.exists():
        return Check(f"{dataset}_pairs", False, f"missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        mapping = raw.get("tasks", raw)
        missing = [task.task_id for task in tasks if task.task_id not in mapping]
        status = raw.get("reference_status", raw.get("annotation_status"))
        empty_references = [
            task.task_id
            for task in tasks
            if task.task_id in mapping
            and not (
                mapping[task.task_id].get("automatic_analogous_ids")
                if status == "automatic"
                else mapping[task.task_id].get("gold_analogous_ids")
            )
        ]
        provenance_ok = status != "automatic" or bool(
            raw.get("reference_provenance", {}).get("algorithm")
        )
        accepted = status in {"automatic", "verified"}
        ok = accepted and provenance_ok and not missing and not empty_references
        detail = (
            f"{path}; reference_status={status}; provenance_ok={provenance_ok}; "
            f"missing_tasks={len(missing)}; empty_references={len(empty_references)}"
        )
        return Check(f"{dataset}_pairs", ok, detail, len(mapping))
    except Exception as exc:
        return Check(f"{dataset}_pairs", False, f"{path}: {type(exc).__name__}: {exc}")
