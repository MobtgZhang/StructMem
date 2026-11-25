import json

from structmem.llm import LLMResponse
from structmem.probe import probe_models


def test_probe_models_requires_four_successes_and_does_not_store_key(tmp_path, monkeypatch) -> None:
    api = tmp_path / "api.json"
    api.write_text(
        json.dumps(
            {
                "base_url": "https://example.test",
                "api_key": "secret-key",
                "models": ["a", "b", "c", "d"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "structmem.probe.LLMClient.complete",
        lambda self, *args, **kwargs: LLMResponse("OK", 2, 1),
    )
    output = tmp_path / "probe.json"
    result = probe_models(api, output)
    assert result["ready"]
    assert len(result["probes"]) == 4
    assert "secret-key" not in output.read_text(encoding="utf-8")


def test_probe_models_accepts_single_model_api_file(tmp_path, monkeypatch) -> None:
    api = tmp_path / "api.json"
    api.write_text(
        json.dumps({"base_url": "https://example.test", "api_key": "secret", "models": ["only"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "structmem.probe.LLMClient.complete",
        lambda self, *args, **kwargs: LLMResponse("OK", 2, 1),
    )
    result = probe_models(api, tmp_path / "probe.json")
    assert result["ready"]
