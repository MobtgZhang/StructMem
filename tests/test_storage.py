import json
from dataclasses import dataclass
from pathlib import Path

from structmem.storage import JSONLStore, load_completed_keys, resolved_episodes, write_json


class NativeBenchmarkObject:
    def __str__(self) -> str:
        return "at(apple, table)"


@dataclass
class NativeRecord:
    name: str


def test_jsonl_store_serializes_benchmark_native_metadata(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    store = JSONLStore(path)

    store.append(
        {
            "facts": [NativeBenchmarkObject()],
            "record": NativeRecord("fact"),
            "labels": {"b", "a"},
            "path": tmp_path,
        }
    )

    assert store.read() == [
        {
            "facts": ["at(apple, table)"],
            "record": {"name": "fact"},
            "labels": ["a", "b"],
            "path": str(tmp_path),
        }
    ]


def test_write_json_uses_same_native_metadata_conversion(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_json(path, {"native": NativeBenchmarkObject()})

    assert json.loads(path.read_text(encoding="utf-8")) == {"native": "at(apple, table)"}


def test_resolved_episodes_prefers_valid_retry_and_keeps_last_unresolved_error() -> None:
    def row(task_id: str, status: str) -> dict:
        return {
            "task": {"dataset": "d", "task_id": task_id},
            "method": "ours",
            "status": status,
        }

    rows = resolved_episodes(
        [
            row("recovered", "agent_error"),
            row("unresolved", "agent_error"),
            row("recovered", "completed"),
            row("unresolved", "infrastructure_error"),
            row("recovered", "agent_error"),
        ]
    )

    indexed = {item["task"]["task_id"]: item for item in rows}
    assert indexed["recovered"]["status"] == "completed"
    assert indexed["unresolved"]["status"] == "infrastructure_error"


def test_load_completed_keys_accepts_versioned_manifest(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text(
        '{"schema_version":1,"keys":[{"dataset":"d","method":"m","task_id":"t"}]}',
        encoding="utf-8",
    )
    assert load_completed_keys(path) == {("d", "m", "t")}
