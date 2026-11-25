from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from structmem.storage import JSONLStore, write_json


def aggregate_four_models(root: Path, output: Path) -> dict[str, Any]:
    runs = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        formal = model_dir / "formal"
        manifest_path = formal / "manifest.json"
        analysis_path = formal / "analysis.json"
        episodes_path = formal / "episodes.jsonl"
        preflight_path = formal / "preflight.json"
        manifest = _read_json(manifest_path)
        analysis = _read_json(analysis_path)
        preflight = _read_json(preflight_path)
        episodes = JSONLStore(episodes_path).read() if episodes_path.exists() else []
        coverage = analysis.get("coverage", [])
        coverage_complete = _coverage_complete(manifest, coverage)
        runs.append(
            {
                "directory": model_dir.name,
                "model": manifest.get("model", model_dir.name),
                "preflight_ready": preflight.get("ready", False),
                "episode_records": len(episodes),
                "valid_episodes": analysis.get("valid_episodes", 0),
                "excluded_errors": analysis.get("excluded_errors", 0),
                "coverage_complete": coverage_complete,
                "coverage": coverage,
                "main_results": analysis.get("main_results", []),
                "rq1_semantic_distance": analysis.get("rq1_semantic_distance", {}),
                "rq2_role_pattern_transfer": analysis.get("rq2_role_pattern_transfer", {}),
                "rq3_component_interactions": analysis.get("rq3_component_interactions", {}),
                "rq4_failure_modes": analysis.get("rq4_failure_modes", {}),
            }
        )
    result = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "models_found": len(runs),
        "complete": len(runs) == 4 and all(
            run["preflight_ready"]
            and run["coverage_complete"]
            and run["excluded_errors"] == 0
            for run in runs
        ),
        "runs": runs,
    }
    write_json(output, result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_complete(manifest: dict[str, Any], coverage: list[dict[str, Any]]) -> bool:
    methods = manifest.get("methods", [])
    datasets = {
        name: int(config.get("tasks", 0))
        for name, config in manifest.get("datasets", {}).items()
        if config.get("enabled", True)
    }
    if not methods or not datasets:
        return False
    indexed = {(row.get("dataset"), row.get("method")): row for row in coverage}
    return all(
        indexed.get((dataset, method), {}).get("valid_n") == tasks
        for dataset, tasks in datasets.items()
        for method in methods
    )
