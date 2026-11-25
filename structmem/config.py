from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_base_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    cache_file: Path | None = None


@dataclass(frozen=True)
class DatasetConfig:
    enabled: bool
    split: str
    tasks: int = 300
    max_steps: int = 50
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    output_dir: Path
    methods: tuple[str, ...]
    datasets: dict[str, DatasetConfig]
    api: APIConfig
    top_k: int = 5
    prefilter_k: int = 50
    memory_token_budget: int = 2000
    resume: bool = True
    memory_file: Path | None = None
    completed_keys_file: Path | None = None
    online_memory_updates: bool = False


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    api_file = _resolve_path(raw["api"]["config_file"], config_path.parent)
    api_raw = json.loads(api_file.read_text(encoding="utf-8"))
    requested_model = os.getenv("STRUCTMEM_MODEL") or raw["api"].get("model")
    api_provider, model = _select_api_provider(api_raw, requested_model)
    api_key = os.getenv("STRUCTMEM_API_KEY", api_provider.get("api_key", api_provider.get("key", "")))
    if not api_key:
        raise ValueError("Missing API key: set STRUCTMEM_API_KEY or api_key in the API config")
    base_url = os.getenv("STRUCTMEM_API_BASE_URL") or api_provider.get("base_url", api_provider.get("url", ""))
    if not base_url:
        raise ValueError(f"Missing base URL for model {model}")
    datasets = {
        name: DatasetConfig(
            enabled=bool(item.get("enabled", True)),
            split=str(item["split"]),
            tasks=int(item.get("tasks", 300)),
            max_steps=int(item.get("max_steps", 50)),
            options=_dataset_options(dict(item.get("options", {})), config_path.parent),
        )
        for name, item in raw["datasets"].items()
    }
    selected_datasets = _environment_selection("STRUCTMEM_DATASETS", tuple(datasets))
    if selected_datasets is not None:
        datasets = {name: datasets[name] for name in selected_datasets}
    methods = tuple(raw["methods"])
    selected_methods = _environment_selection("STRUCTMEM_METHODS", methods)
    if selected_methods is not None:
        methods = selected_methods
    memory_value = os.getenv("STRUCTMEM_MEMORY_FILE") or raw.get("memory_file")
    completed_keys_value = os.getenv("STRUCTMEM_COMPLETED_KEYS_FILE") or raw.get("completed_keys_file")
    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        output_dir=_resolve_path(
            os.getenv("STRUCTMEM_OUTPUT_DIR", raw["output_dir"]),
            config_path.parent,
        ),
        methods=methods,
        datasets=datasets,
        api=APIConfig(
            base_url=normalize_base_url(str(base_url)),
            api_key=api_key,
            model=model,
            timeout_seconds=float(raw["api"].get("timeout_seconds", 120)),
            max_retries=int(raw["api"].get("max_retries", 3)),
            retry_base_seconds=float(raw["api"].get("retry_base_seconds", 5)),
            retry_max_seconds=float(raw["api"].get("retry_max_seconds", 60)),
            cache_file=(
                _resolve_path(
                    os.getenv("STRUCTMEM_LLM_CACHE_FILE", raw["api"].get("cache_file")),
                    config_path.parent,
                )
                if os.getenv("STRUCTMEM_LLM_CACHE_FILE", raw["api"].get("cache_file"))
                else None
            ),
        ),
        top_k=int(raw.get("top_k", 5)),
        prefilter_k=int(raw.get("prefilter_k", 50)),
        memory_token_budget=int(raw.get("memory_token_budget", 2000)),
        resume=bool(raw.get("resume", True)),
        memory_file=_resolve_path(memory_value, config_path.parent) if memory_value else None,
        completed_keys_file=(
            _resolve_path(completed_keys_value, config_path.parent)
            if completed_keys_value
            else None
        ),
        online_memory_updates=bool(raw.get("online_memory_updates", False)),
    )


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _select_api_provider(
    api_raw: dict[str, Any],
    requested_model: str | None,
) -> tuple[dict[str, Any], str]:
    providers = api_raw.get("all")
    if not isinstance(providers, dict):
        models = api_raw.get("models", [])
        model = requested_model or (str(models[0]) if models else "")
        if not model:
            raise ValueError("API config must define at least one model")
        return api_raw, model

    selected = api_raw.get("selected", [])
    if not isinstance(selected, list) or not selected:
        raise ValueError("Multi-provider API config must define a non-empty selected model list")
    model = requested_model or str(selected[0])
    if model not in selected:
        raise ValueError(f"Model {model!r} is not selected; selected={selected}")
    matches = [
        provider
        for provider in providers.values()
        if isinstance(provider, dict) and model in provider.get("models", [])
    ]
    if not matches:
        raise ValueError(f"No API provider defines selected model {model!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple API providers define selected model {model!r}")
    return matches[0], model


def _environment_selection(name: str, available: tuple[str, ...]) -> tuple[str, ...] | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    selected = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not selected:
        raise ValueError(f"{name} must contain at least one comma-separated value")
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(f"{name} contains unknown values {unknown}; available={list(available)}")
    return selected


def _dataset_options(options: dict[str, Any], base: Path) -> dict[str, Any]:
    resolved = dict(options)
    if resolved.get("pair_manifest"):
        pair_dir = os.getenv("STRUCTMEM_PAIR_DIR")
        pair_value = (
            str(Path(pair_dir) / Path(str(resolved["pair_manifest"])).name)
            if pair_dir
            else str(resolved["pair_manifest"])
        )
        resolved["pair_manifest"] = str(_resolve_path(pair_value, base))
    if resolved.get("config"):
        resolved["config"] = str(_resolve_path(str(resolved["config"]), base))
    return resolved
