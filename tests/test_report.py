import json

from structmem.report import aggregate_four_models


def test_aggregate_preserves_coverage_and_does_not_claim_incomplete_run(tmp_path) -> None:
    model = tmp_path / "m" / "formal"
    model.mkdir(parents=True)
    (model / "manifest.json").write_text(json.dumps({"model": "m"}), encoding="utf-8")
    (model / "preflight.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (model / "analysis.json").write_text(
        json.dumps(
            {
                "valid_episodes": 1,
                "excluded_errors": 2,
                "coverage": [{"dataset": "d", "method": "x", "valid_n": 1}],
                "main_results": [],
            }
        ),
        encoding="utf-8",
    )
    (model / "episodes.jsonl").write_text(
        json.dumps({"status": "completed"}) + "\n",
        encoding="utf-8",
    )
    result = aggregate_four_models(tmp_path, tmp_path / "aggregate.json")
    assert not result["complete"]
    assert result["runs"][0]["coverage"][0]["valid_n"] == 1


def test_aggregate_requires_exact_manifest_coverage_and_no_errors(tmp_path) -> None:
    for name in ("a", "b", "c", "d"):
        formal = tmp_path / name / "formal"
        formal.mkdir(parents=True)
        (formal / "manifest.json").write_text(
            json.dumps(
                {
                    "model": name,
                    "methods": ["ours"],
                    "datasets": {"agentbench": {"enabled": True, "tasks": 300}},
                }
            ),
            encoding="utf-8",
        )
        (formal / "preflight.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
        (formal / "analysis.json").write_text(
            json.dumps(
                {
                    "valid_episodes": 300,
                    "excluded_errors": 0,
                    "coverage": [{"dataset": "agentbench", "method": "ours", "valid_n": 300}],
                }
            ),
            encoding="utf-8",
        )
    result = aggregate_four_models(tmp_path, tmp_path / "aggregate.json")
    assert result["complete"]
    assert all(run["coverage_complete"] for run in result["runs"])

    analysis = tmp_path / "a" / "formal" / "analysis.json"
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    payload["excluded_errors"] = 1
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    assert not aggregate_four_models(tmp_path, tmp_path / "aggregate.json")["complete"]
