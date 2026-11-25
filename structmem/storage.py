from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class JSONLStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: dict[str, Any]) -> None:
        line = json.dumps(item, ensure_ascii=True, sort_keys=True, default=_json_default)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSONL at {self.path}:{line_number}") from exc
        return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    temp.replace(path)


def _json_default(value: Any) -> Any:
    """Convert benchmark-native metadata without changing its in-memory representation."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


def manifest(config_path: Path, config: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": config.seed,
        "methods": list(config.methods),
        "datasets": {
            name: {
                "enabled": item.enabled,
                "split": item.split,
                "tasks": item.tasks,
                "max_steps": item.max_steps,
                "options": item.options,
            }
            for name, item in config.datasets.items()
        },
        "model": config.api.model,
        "base_url": config.api.base_url,
        "top_k": config.top_k,
        "prefilter_k": config.prefilter_k,
        "memory_token_budget": config.memory_token_budget,
        "memory_file": str(config.memory_file) if config.memory_file else None,
        "completed_keys_file": (
            str(config.completed_keys_file) if config.completed_keys_file else None
        ),
        "online_memory_updates": config.online_memory_updates,
    }


def completed_keys(items: Iterable[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (item["task"]["dataset"], item["method"], item["task"]["task_id"])
        for item in items
        if item.get("status") in {"completed", "task_failed"}
    }


def load_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("keys", []) if isinstance(payload, dict) else payload
    result = set()
    for row in rows:
        if isinstance(row, dict):
            key = (row.get("dataset"), row.get("method"), row.get("task_id"))
        else:
            key = tuple(row) if isinstance(row, list) else ()
        if len(key) != 3 or not all(isinstance(item, str) and item for item in key):
            raise ValueError(f"Invalid completed key in {path}: {row!r}")
        result.add(key)
    return result


def resolved_episodes(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve retry attempts to one record per dataset, method, and task.

    A later valid result supersedes earlier infrastructure/agent errors. If no
    attempt is valid, the last error remains visible for diagnosis.
    """
    resolved: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in items:
        task = item.get("task", {})
        key = (str(task.get("dataset", "")), str(item.get("method", "")), str(task.get("task_id", "")))
        previous = resolved.get(key)
        valid = item.get("status") in {"completed", "task_failed"}
        previous_valid = previous is not None and previous.get("status") in {"completed", "task_failed"}
        if previous is None or valid or not previous_valid:
            resolved[key] = item
    return list(resolved.values())
