import json

import pytest

from structmem.config import load_config


def _write_experiment(tmp_path, api_file, model):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"""
seed: 42
output_dir: results
api:
  config_file: {api_file.name}
  model: {model}
methods: [no_memory]
datasets:
  agentbench:
    enabled: true
    split: standard
    tasks: 1
    max_steps: 1
""",
        encoding="utf-8",
    )
    return config


def test_load_config_selects_model_provider_from_multi_provider_api(tmp_path) -> None:
    api = tmp_path / "api.json"
    api.write_text(
        json.dumps(
            {
                "all": {
                    "first": {"url": "https://first.test", "key": "first-key", "models": ["alpha"]},
                    "second": {"url": "https://second.test", "key": "second-key", "models": ["beta"]},
                },
                "selected": ["alpha", "beta"],
            }
        ),
        encoding="utf-8",
    )

    config = load_config(_write_experiment(tmp_path, api, "beta"))

    assert config.api.model == "beta"
    assert config.api.base_url == "https://second.test/v1"
    assert config.api.api_key == "second-key"


def test_load_config_rejects_model_outside_selected_list(tmp_path) -> None:
    api = tmp_path / "api.json"
    api.write_text(
        json.dumps(
            {
                "all": {
                    "provider": {
                        "url": "https://provider.test",
                        "key": "key",
                        "models": ["selected", "not-selected"],
                    }
                },
                "selected": ["selected"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not selected"):
        load_config(_write_experiment(tmp_path, api, "not-selected"))
