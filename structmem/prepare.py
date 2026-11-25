from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from structmem.agents.baselines import episode_to_experience
from structmem.llm import LLMClient, LLMError
from structmem.matrix import build_matrix
from structmem.memory import ExperienceStore
from structmem.storage import JSONLStore, resolved_episodes, write_json
from structmem.types import Episode, RunStatus, Step, TaskSpec


def memory_from_episodes(input_path: Path, output_path: Path, *, include_failures: bool = False) -> dict[str, Any]:
    raw_records = JSONLStore(input_path).read()
    raw = resolved_episodes(raw_records)
    store = ExperienceStore()
    skipped = 0
    for item in raw:
        valid_status = item.get("status") in {"completed", "task_failed"}
        if not valid_status or (not include_failures and not item.get("success")):
            skipped += 1
            continue
        episode = _episode(item)
        lesson = "Successful training trajectory" if episode.success else "Failed training trajectory; avoid copying blindly"
        experience = episode_to_experience(episode, lesson=lesson)
        store.add(experience)
    store.save(output_path)
    return {
        "input": len(raw_records),
        "resolved": len(raw),
        "saved": len(store.values()),
        "successes": sum(item.success for item in store.values()),
        "failures": sum(not item.success for item in store.values()),
        "skipped": skipped,
        "output": str(output_path),
    }


def queries_from_episodes(input_path: Path, output_path: Path) -> dict[str, Any]:
    raw_records = JSONLStore(input_path).read()
    raw = resolved_episodes(raw_records)
    store = ExperienceStore()
    skipped = 0
    for item in raw:
        if item.get("status") not in {"completed", "task_failed"}:
            skipped += 1
            continue
        episode = _episode(item)
        experience = episode_to_experience(episode, lesson="Held-out query trace for matrix annotation")
        experience.experience_id = f"query:{episode.task.dataset}:{episode.task.task_id}"
        store.add(experience)
    store.save(output_path)
    return {
        "input": len(raw_records),
        "resolved": len(raw),
        "saved": len(store.values()),
        "skipped": skipped,
        "output": str(output_path),
    }


def matrix_from_memories(
    query_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    semantic_threshold: float,
    structural_threshold: float,
    per_cell: int,
    seed: int,
) -> dict[str, Any]:
    queries = ExperienceStore.load(query_path).values()
    sources = ExperienceStore.load(source_path).values()
    data = build_matrix(
        queries,
        sources,
        semantic_threshold=semantic_threshold,
        structural_threshold=structural_threshold,
        per_cell=per_cell,
        seed=seed,
    )
    write_json(output_path, data)
    return data


def automatic_manifests_from_memories(
    query_path: Path,
    source_path: Path,
    output_dir: Path,
    *,
    semantic_threshold: float,
    structural_threshold: float,
    per_cell: int,
    seed: int,
) -> dict[str, Any]:
    queries = ExperienceStore.load(query_path).values()
    sources = ExperienceStore.load(source_path).values()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset in sorted({item.dataset for item in queries}):
        target_queries = [item for item in queries if item.dataset == dataset]
        data = build_matrix(
            target_queries,
            sources,
            semantic_threshold=semantic_threshold,
            structural_threshold=structural_threshold,
            per_cell=per_cell,
            seed=seed,
        )
        data["dataset"] = dataset
        path = output_dir / f"{dataset}.json"
        write_json(path, data)
        results[dataset] = {
            "queries": len(target_queries),
            "manifest_tasks": len(data["tasks"]),
            "output": str(path),
        }
    return {
        "reference_status": "automatic",
        "source_experiences": len(sources),
        "datasets": results,
    }


FAILURE_CATEGORIES = (
    "incomplete_binding",
    "hidden_preconditions",
    "cascading_errors",
    "ambiguous_role_assignment",
    "other",
)
FAILURE_AUDIT_PROMPT_VERSION = "rq4_failure_judge_v1"


def failure_audit_from_episodes(
    input_paths: list[Path],
    output_path: Path,
    *,
    llm: LLMClient,
    limit: int = 50,
    observation_chars: int = 700,
) -> dict[str, Any]:
    records = []
    for input_path in input_paths:
        records.extend(JSONLStore(input_path).read())
    raw = resolved_episodes(records)
    candidates = []
    for item in raw:
        if (
            item.get("status") not in {"completed", "task_failed"}
            or item.get("method") != "ours"
            or item.get("success")
            or not item.get("transfer_used")
        ):
            continue
        diagnostics = item.get("diagnostics", {})
        scores = diagnostics.get("ranked_scores", {})
        retrieved_ids = item.get("retrieved_ids", [])
        structural_score = max((float(scores.get(key, 0.0)) for key in retrieved_ids), default=0.0)
        candidates.append((structural_score, item))
    candidates.sort(
        key=lambda pair: (
            -pair[0],
            pair[1]["task"]["dataset"],
            pair[1]["task"]["task_id"],
        )
    )
    selected = candidates[:limit]
    items = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    classification_errors = []
    for structural_score, episode in selected:
        task = episode["task"]
        diagnostics = episode.get("diagnostics", {})
        failure_id = f"{task['dataset']}:{task['task_id']}"
        evidence = {
            "failure_id": failure_id,
            "dataset": task["dataset"],
            "goal": task.get("goal", ""),
            "max_retrieved_structural_score": structural_score,
            "retrieved_ids": episode.get("retrieved_ids", []),
            "binding_outcome": episode.get("binding_outcome"),
            "gate_verdict": episode.get("gate_verdict"),
            "binding_reason": diagnostics.get("binding_reason", ""),
            "actions": [step.get("action", "") for step in episode.get("steps", [])][-20:],
            "observations": [
                str(step.get("observation", ""))[:observation_chars]
                for step in episode.get("steps", [])[-20:]
            ],
        }
        try:
            payload, response = llm.json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Classify one failed interactive-agent transfer using only the supplied trace. "
                            "Choose exactly one category: incomplete_binding (an intended role could not be "
                            "grounded to a valid action), hidden_preconditions (an apparently valid action was "
                            "blocked by state or resource constraints), cascading_errors (an earlier mistake "
                            "corrupted later steps), ambiguous_role_assignment (the same action or observation "
                            "supports competing functional roles), or other. Return JSON with category, confidence "
                            "from 0 to 1, and a concrete one-sentence explanation. Do not claim human verification."
                        ),
                    },
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=True)},
                ],
                max_tokens=260,
            )
            category = str(payload.get("category", "other"))
            if category not in FAILURE_CATEGORIES:
                category = "other"
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
            explanation = str(payload.get("explanation", "")).strip()
            if not explanation:
                raise ValueError("Classifier returned an empty explanation")
            usage = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        except (LLMError, TypeError, ValueError) as exc:
            category = "other"
            confidence = 0.0
            explanation = f"Automatic classifier failed: {type(exc).__name__}: {exc}"
            usage = {"input_tokens": 0, "output_tokens": 0}
            classification_errors.append(failure_id)
        for key in total_usage:
            total_usage[key] += usage[key]
        items.append(
            {
                **evidence,
                "task_id": task["task_id"],
                "category": category,
                "confidence": confidence,
                "explanation": explanation,
                "judge_model": llm.config.model,
                "prompt_version": FAILURE_AUDIT_PROMPT_VERSION,
                "token_usage": usage,
            }
        )
    data = {
        "schema_version": 2,
        "annotation_status": "automatic_complete" if not classification_errors else "automatic_partial",
        "categories": list(FAILURE_CATEGORIES),
        "provenance": {
            "classifier": "llm",
            "model": llm.config.model,
            "prompt_version": FAILURE_AUDIT_PROMPT_VERSION,
            "human_annotation": False,
            "warning": "Model-assisted diagnostic labels are not human ground truth.",
            "input_files": [str(path) for path in input_paths],
        },
        "requested": limit,
        "eligible": len(candidates),
        "selected": len(items),
        "classification_errors": classification_errors,
        "token_usage": total_usage,
        "items": items,
    }
    write_json(output_path, data)
    return data


def _episode(item: dict[str, Any]) -> Episode:
    task = TaskSpec(**item["task"])
    steps = [Step(**step) for step in item.get("steps", [])]
    return Episode(
        task=task,
        method=item["method"],
        status=RunStatus(item["status"]),
        success=bool(item["success"]),
        score=float(item["score"]),
        steps=steps,
        error=item.get("error"),
    )
