from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from structmem.config import APIConfig, normalize_base_url
from structmem.llm import LLMClient
from structmem.storage import write_json


def probe_models(api_path: Path, output: Path) -> dict[str, Any]:
    raw = json.loads(api_path.read_text(encoding="utf-8"))
    models = list(raw.get("models", []))
    api_key = os.getenv("STRUCTMEM_API_KEY", raw.get("api_key", ""))
    if not api_key:
        raise ValueError("Missing API key: set STRUCTMEM_API_KEY or api_key in the API config")
    base_url = normalize_base_url(os.getenv("STRUCTMEM_API_BASE_URL") or raw["base_url"])
    probes = []
    for model in models:
        started = time.monotonic()
        try:
            response = LLMClient(
                APIConfig(
                    base_url=base_url,
                    api_key=api_key,
                    model=str(model),
                    timeout_seconds=30,
                    max_retries=1,
                )
            ).complete(
                [{"role": "user", "content": "Reply with OK."}],
                max_tokens=8,
            )
            probes.append(
                {
                    "model": model,
                    "ok": True,
                    "latency_seconds": time.monotonic() - started,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }
            )
        except Exception as exc:
            probes.append(
                {
                    "model": model,
                    "ok": False,
                    "latency_seconds": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    result = {
        "schema_version": 1,
        "base_url": base_url,
        "models_expected": len(models),
        "ready": bool(models) and all(item["ok"] for item in probes),
        "probes": probes,
    }
    write_json(output, result)
    return result
