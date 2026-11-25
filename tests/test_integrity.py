from pathlib import Path

import json
import pytest

from structmem.environments.base import InfrastructureError
from structmem.config import load_config
from structmem.runner import ExperimentRunner
from structmem.types import TaskSpec


def test_no_legacy_import_hacks() -> None:
    root = Path(__file__).parents[1] / "structmem"
    forbidden = ("sys.path", "importlib", "old-codes", "old_codes", "SourceFileLoader")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token!r} found in {path}"


def test_runner_rejects_empty_human_gold(tmp_path, monkeypatch) -> None:
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {
                "annotation_status": "verified",
                "tasks": {"t": {"cell": "far_semantic_near_structural", "gold_analogous_ids": []}},
            }
        ),
        encoding="utf-8",
    )
    runner = object.__new__(ExperimentRunner)
    runner.config_path = tmp_path / "config.yaml"
    with pytest.raises(InfrastructureError, match="without analogous references"):
        runner._attach_pair_manifest([TaskSpec("d", "t", "goal", "test", 0)], str(path))


def test_runner_accepts_provenanced_automatic_references(tmp_path) -> None:
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {
                "reference_status": "automatic",
                "reference_provenance": {"algorithm": "frozen-test-v1"},
                "tasks": {
                    "t": {
                        "cell": "far_semantic_near_structural",
                        "automatic_analogous_ids": ["source"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runner = object.__new__(ExperimentRunner)
    runner.config_path = tmp_path / "config.yaml"
    attached = runner._attach_pair_manifest(
        [TaskSpec("d", "t", "goal", "test", 0)], str(path)
    )
    assert attached[0].metadata["analogous_reference_ids"] == ["source"]
    assert attached[0].metadata["reference_status"] == "automatic"


def test_config_environment_overrides_isolate_model_outputs(tmp_path, monkeypatch) -> None:
    api = tmp_path / "api.json"
    api.write_text(
        json.dumps(
            {
                "base_url": "https://example.invalid",
                "api_key": "test-key",
                "models": ["default-model"],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
seed: 42
output_dir: default-output
memory_file: default-memory.json
methods: [no_memory, ours]
api:
  config_file: api.json
datasets:
  alfworld:
    enabled: false
    split: train
    options:
      pair_manifest: default-pairs/alfworld.json
  agentbench:
    enabled: true
    split: standard
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STRUCTMEM_MODEL", "isolated-model")
    monkeypatch.setenv("STRUCTMEM_OUTPUT_DIR", str(tmp_path / "isolated-output"))
    monkeypatch.setenv("STRUCTMEM_MEMORY_FILE", str(tmp_path / "isolated-memory.json"))
    monkeypatch.setenv("STRUCTMEM_PAIR_DIR", str(tmp_path / "isolated-pairs"))
    monkeypatch.setenv("STRUCTMEM_METHODS", "ours")
    monkeypatch.setenv("STRUCTMEM_DATASETS", "agentbench")
    loaded = load_config(config)
    assert loaded.api.model == "isolated-model"
    assert loaded.output_dir == tmp_path / "isolated-output"
    assert loaded.memory_file == tmp_path / "isolated-memory.json"
    assert loaded.methods == ("ours",)
    assert tuple(loaded.datasets) == ("agentbench",)
