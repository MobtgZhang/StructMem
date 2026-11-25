from __future__ import annotations

from collections import defaultdict
from typing import Any

from structmem.types import Cell
from structmem.storage import resolved_episodes


VALID_STATUSES = {"completed", "task_failed"}
FACTORIAL_METHODS = {
    "ours": (1, 1, 1),
    "w_o_roles": (0, 1, 1),
    "w_o_structure": (1, 0, 1),
    "w_o_binding": (1, 1, 0),
    "w_o_roles_structure": (0, 0, 1),
    "w_o_roles_binding": (0, 1, 0),
    "w_o_structure_binding": (1, 0, 0),
    "w_o_roles_structure_binding": (0, 0, 0),
}
FAILURE_CATEGORIES = {
    "incomplete_binding",
    "hidden_preconditions",
    "cascading_errors",
    "ambiguous_role_assignment",
    "other",
}


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def analyze(
    episodes: list[dict[str, Any]],
    failure_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_episode_records = len(episodes)
    episodes = resolved_episodes(episodes)
    valid = [episode for episode in episodes if episode.get("status") in VALID_STATUSES]
    errors = [episode for episode in episodes if episode.get("status") not in VALID_STATUSES]
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in valid:
        metadata = episode["task"].get("metadata", {})
        cell = metadata.get("cell", "unassigned")
        grouped[(episode["task"]["dataset"], episode["method"], cell)].append(episode)
    main = []
    for (dataset, method, cell), items in sorted(grouped.items()):
        main.append(
            {
                "dataset": dataset,
                "method": method,
                "cell": cell,
                "n": len(items),
                "success_rate": _rate([bool(item["success"]) for item in items]),
                "mean_score": sum(float(item["score"]) for item in items) / len(items),
                "mean_steps": sum(len(item["steps"]) for item in items) / len(items),
            }
        )
    rq1 = _rq1_semantic_distance(valid)
    rq2 = _rq2_role_patterns(valid)
    rq3 = _rq3_factorial(episodes)
    rq4 = _rq4_failure_modes(valid, failure_audit)
    return {
        "schema_version": 2,
        "raw_episode_records": raw_episode_records,
        "resolved_episode_records": len(episodes),
        "valid_episodes": len(valid),
        "excluded_errors": len(errors),
        "error_counts": _error_counts(errors),
        "error_breakdown": _error_breakdown(errors),
        "coverage": _coverage(episodes),
        "main_results": main,
        "rq1_semantic_distance": rq1,
        "rq2_role_pattern_transfer": rq2,
        "rq3_component_interactions": rq3,
        "rq4_failure_modes": rq4,
        "supplementary_diagnostics": _supplementary(grouped, valid),
    }


def _rq1_semantic_distance(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    records: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in episodes:
        if item["method"] not in {"ours", "embedding_retrieval"}:
            continue
        scores = item.get("diagnostics", {}).get("retrieved_semantic_scores", {})
        if not scores:
            continue
        similarity = sum(float(value) for value in scores.values()) / len(scores)
        records[(item["task"]["dataset"], item["method"])].append(
            {
                "task_id": item["task"]["task_id"],
                "semantic_similarity": similarity,
                "semantic_distance": 1.0 - similarity,
                "success_at_7": bool(item["success"]) and len(item.get("steps", [])) <= 7,
            }
        )
    quartiles = []
    for (dataset, method), items in sorted(records.items()):
        ordered = sorted(items, key=lambda row: (row["semantic_distance"], row["task_id"]))
        bins: list[list[dict[str, Any]]] = [[] for _ in range(4)]
        for rank, row in enumerate(ordered):
            index = min(3, rank * 4 // len(ordered))
            bins[index].append(row)
        for index, rows in enumerate(bins, 1):
            quartiles.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "quartile": index,
                    "label": "near" if index == 1 else "far" if index == 4 else "middle",
                    "n": len(rows),
                    "mean_semantic_distance": (
                        sum(row["semantic_distance"] for row in rows) / len(rows) if rows else None
                    ),
                    "success_rate_at_7_steps": _rate([row["success_at_7"] for row in rows]),
                }
            )
    complexity = []
    by_key = {
        (item["task"]["dataset"], item["task"]["task_id"], item["method"]): item
        for item in episodes
    }
    for dataset in sorted({key[0] for key in by_key}):
        pairs = []
        node_counts = []
        edge_counts = []
        for (item_dataset, task_id, method), structural in by_key.items():
            if item_dataset != dataset or method != "ours":
                continue
            embedding = by_key.get((dataset, task_id, "embedding_retrieval"))
            if embedding is None:
                continue
            diagnostics = structural.get("diagnostics", {})
            node_counts.append(int(diagnostics.get("query_node_count", 0)))
            edge_counts.append(int(diagnostics.get("query_edge_count", 0)))
            pairs.append((bool(structural["success"]), bool(embedding["success"])))
        if pairs:
            structural_rate = _rate([left for left, _ in pairs])
            embedding_rate = _rate([right for _, right in pairs])
            complexity.append(
                {
                    "dataset": dataset,
                    "paired_n": len(pairs),
                    "mean_query_nodes": sum(node_counts) / len(node_counts),
                    "mean_query_edges": sum(edge_counts) / len(edge_counts),
                    "structural_success_rate": structural_rate,
                    "embedding_success_rate": embedding_rate,
                    "structural_advantage_pp": 100.0 * (structural_rate - embedding_rate),
                }
            )
    return {
        "definition": "Paper RQ1: success within 7 steps stratified by retrieved semantic-distance quartile.",
        "quartiles": quartiles,
        "benchmark_complexity": complexity,
        "ready": bool(quartiles) and bool(complexity),
    }


def _rq2_role_patterns(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    patterns: defaultdict[tuple[str, str, str], list[tuple[bool, bool]]] = defaultdict(list)
    for item in episodes:
        if item["method"] != "ours" or not item.get("transfer_used"):
            continue
        target_dataset = item["task"]["dataset"]
        retrieved = item.get("diagnostics", {}).get("retrieved_metadata", [])
        seen_in_episode: dict[tuple[str, str, str], bool] = {}
        for source in retrieved:
            cross_benchmark = source.get("dataset") != target_dataset
            roles = [str(role) for role in source.get("roles", [])]
            for index in range(len(roles) - 2):
                pattern = tuple(roles[index : index + 3])
                seen_in_episode[pattern] = seen_in_episode.get(pattern, False) or cross_benchmark
        for pattern, cross_benchmark in seen_in_episode.items():
            patterns[pattern].append((bool(item["success"]), cross_benchmark))
    rows = [
        {
            "pattern": list(pattern),
            "label": " -> ".join(pattern),
            "attempts": len(exposures),
            "successful_transfers": sum(success for success, _ in exposures),
            "transfer_success_rate": _rate([success for success, _ in exposures]),
            "cross_domain_transfer_rate": _rate([success for success, _ in exposures]),
            "cross_benchmark_exposures": sum(cross for _, cross in exposures),
        }
        for pattern, exposures in patterns.items()
    ]
    rows.sort(key=lambda row: (-row["transfer_success_rate"], -row["attempts"], row["label"]))
    return {
        "definition": "Paper RQ2: episode success rate after exposure to each retrieved source-role trigram.",
        "scope": "Benchmark-scoped unless cross_benchmark_exposures is nonzero.",
        "patterns": rows,
        "ready": bool(rows),
    }


def _rq3_factorial(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: defaultdict[tuple[str, str], dict[tuple[int, int, int], bool]] = defaultdict(dict)
    for item in episodes:
        factors = FACTORIAL_METHODS.get(item["method"])
        if factors is not None:
            key = (item["task"]["dataset"], item["task"]["task_id"])
            by_task[key]
            if item.get("status") in VALID_STATUSES:
                by_task[key][factors] = bool(item["success"])
    rows = []
    expected = set(FACTORIAL_METHODS.values())
    datasets = sorted({dataset for dataset, _ in by_task})
    missing = []
    for dataset in datasets:
        complete = [cells for (item_dataset, _), cells in by_task.items() if item_dataset == dataset and set(cells) == expected]
        incomplete = [
            task_id
            for (item_dataset, task_id), cells in by_task.items()
            if item_dataset == dataset and set(cells) != expected
        ]
        if incomplete:
            missing.append({"dataset": dataset, "incomplete_tasks": len(incomplete), "first_task": incomplete[0]})
        for left, right, fixed_index, label in (
            (0, 1, 2, "Roles x Structure"),
            (0, 2, 1, "Roles x Binding"),
            (1, 2, 0, "Structure x Binding"),
        ):
            task_effects = []
            for cells in complete:
                fixed_contrasts = []
                for fixed in (0, 1):
                    values = {}
                    for first in (0, 1):
                        for second in (0, 1):
                            factors = [0, 0, 0]
                            factors[left] = first
                            factors[right] = second
                            factors[fixed_index] = fixed
                            values[(first, second)] = float(cells[tuple(factors)])
                    fixed_contrasts.append(
                        values[(1, 1)] - values[(1, 0)] - values[(0, 1)] + values[(0, 0)]
                    )
                task_effects.append(sum(fixed_contrasts) / 2.0)
            rows.append(
                {
                    "dataset": dataset,
                    "interaction": label,
                    "paired_n": len(task_effects),
                    "effect_pp": 100.0 * sum(task_effects) / len(task_effects) if task_effects else None,
                }
            )
    return {
        "definition": "Paper RQ3: pairwise second-difference effects in the complete Roles x Structure x Binding design.",
        "factor_order": ["roles", "structure", "binding"],
        "method_cells": {method: list(cell) for method, cell in FACTORIAL_METHODS.items()},
        "interactions": rows,
        "missing": missing,
        "ready": bool(rows) and all(row["paired_n"] > 0 for row in rows) and not missing,
    }


def _rq4_failure_modes(
    episodes: list[dict[str, Any]],
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    eligible_ids = {
        f"{item['task']['dataset']}:{item['task']['task_id']}"
        for item in episodes
        if item["method"] == "ours"
        and bool(item.get("transfer_used"))
        and not bool(item["success"])
    }
    eligible = len(eligible_ids)
    if audit is None:
        return {
            "definition": "Paper RQ4: provenance-aware categorization of failed structural-transfer attempts.",
            "ready": False,
            "eligible_failures": eligible,
            "detail": "No failure audit supplied. Run build-failure-audit with the experiment config.",
            "categories": [],
        }
    items = audit.get("items", [])
    annotation_status = audit.get("annotation_status")
    manual_verified = annotation_status == "verified"
    automatic_complete = annotation_status == "automatic_complete"
    provenance = audit.get("provenance", {})
    provenance_ok = manual_verified or (
        automatic_complete
        and provenance.get("classifier") == "llm"
        and provenance.get("model")
        and provenance.get("prompt_version")
        and provenance.get("human_annotation") is False
    )
    invalid = [item.get("failure_id") for item in items if item.get("category") not in FAILURE_CATEGORIES]
    ids = [item.get("failure_id") for item in items]
    duplicate_ids = sorted({failure_id for failure_id in ids if ids.count(failure_id) > 1})
    unknown_ids = sorted(failure_id for failure_id in ids if failure_id not in eligible_ids)
    missing_notes = [
        item.get("failure_id")
        for item in items
        if not str(item.get("annotator_note") or item.get("explanation") or "").strip()
    ]
    invalid_confidence = [
        item.get("failure_id")
        for item in items
        if automatic_complete
        and not isinstance(item.get("confidence"), (int, float))
    ]
    required_audit_size = min(50, eligible)
    ready = (
        provenance_ok
        and required_audit_size > 0
        and len(items) >= required_audit_size
        and not invalid
        and not duplicate_ids
        and not unknown_ids
        and not missing_notes
        and not invalid_confidence
    )
    counts: defaultdict[str, int] = defaultdict(int)
    if ready:
        for item in items:
            counts[item["category"]] += 1
    return {
        "definition": "Paper RQ4: provenance-aware categorization of failed structural-transfer attempts.",
        "ready": ready,
        "eligible_failures": eligible,
        "required_audit_size": required_audit_size,
        "audited_n": len(items),
        "annotation_status": annotation_status,
        "manual_verified": manual_verified,
        "automatic_complete": automatic_complete,
        "provenance": provenance,
        "invalid_or_unlabeled": invalid,
        "duplicate_failure_ids": duplicate_ids,
        "unknown_failure_ids": unknown_ids,
        "missing_annotator_notes": missing_notes,
        "invalid_confidence": invalid_confidence,
        "categories": [
            {
                "category": category,
                "count": counts.get(category, 0),
                "fraction": counts.get(category, 0) / len(items) if ready else None,
            }
            for category in sorted(FAILURE_CATEGORIES)
        ],
    }


def _supplementary(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = _cell_success(grouped, Cell.FAR_NEAR.value)
    false_analogy = _false_analogy(grouped)
    gate = _gate_effectiveness(episodes)
    retrieval = _retrieval_accuracy(episodes)
    return {
        "positive_transfer": positive,
        "false_analogy": false_analogy,
        "gate_effectiveness": gate,
        "retrieval_accuracy": retrieval,
        "positive_transfer_structural_vs_embedding": _contrast(
            positive, "ours", "embedding_retrieval", "success_rate"
        ),
        "false_analogy_structural_vs_embedding": _contrast(
            false_analogy, "ours", "embedding_retrieval", "false_transfer_rate", reverse=True
        ),
        "gate_vs_unguarded": _contrast(
            gate, "ours", "w_o_gate", "negative_transfer_rate", reverse=True
        ),
        "retrieval_structural_vs_embedding": _contrast(
            retrieval, "ours", "embedding_retrieval", "top5_hit_rate"
        ),
    }


def _cell_success(grouped: dict[tuple[str, str, str], list[dict[str, Any]]], cell: str) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "method": method,
            "n": len(items),
            "success_rate": _rate([bool(item["success"]) for item in items]),
        }
        for (dataset, method, group_cell), items in sorted(grouped.items())
        if group_cell == cell
    ]


def _false_analogy(grouped: dict[tuple[str, str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for (dataset, method, cell), items in sorted(grouped.items()):
        if cell != Cell.NEAR_FAR.value:
            continue
        result.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(items),
                "rejection_rate": _rate([item.get("gate_verdict") == "skip" for item in items]),
                "false_transfer_rate": _rate(
                    [bool(item.get("transfer_used")) and not bool(item["success"]) for item in items]
                ),
            }
        )
    return result


def _gate_effectiveness(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relevant = {Cell.NEAR_FAR.value, Cell.FAR_FAR.value}
    by_key = {
        (item["task"]["dataset"], item["task"]["task_id"], item["method"]): item
        for item in episodes
        if item["task"].get("metadata", {}).get("cell") in relevant
    }
    result = []
    for dataset in sorted({key[0] for key in by_key}):
        for method in ("ours", "w_o_gate"):
            harms = []
            for (item_dataset, task_id, item_method), item in by_key.items():
                if item_dataset == dataset and item_method == method:
                    control = by_key.get((dataset, task_id, "no_memory"))
                    if control is not None:
                        harms.append(bool(control["success"]) and not bool(item["success"]))
            result.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "paired_n": len(harms),
                    "negative_transfer_rate": _rate(harms),
                }
            )
    return result


def _retrieval_accuracy(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for item in episodes:
        metadata = item["task"].get("metadata", {})
        references = set(
            metadata.get("analogous_reference_ids", metadata.get("gold_analogous_ids", []))
        )
        if references:
            reference_status = metadata.get("reference_status", "verified")
            grouped[(item["task"]["dataset"], item["method"], reference_status)].append(
                bool(references & set(item.get("retrieved_ids", [])[:5]))
            )
    return [
        {
            "dataset": dataset,
            "method": method,
            "reference_status": reference_status,
            "metric": "top5_pseudo_reference_hit_rate" if reference_status == "automatic" else "top5_hit_rate",
            "n": len(hits),
            "top5_hit_rate": _rate(hits),
        }
        for (dataset, method, reference_status), hits in sorted(grouped.items())
    ]


def _error_counts(errors: list[dict[str, Any]]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for item in errors:
        counts[item.get("status", "unknown")] += 1
    return dict(sorted(counts.items()))


def _error_breakdown(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for item in errors:
        counts[(item["task"]["dataset"], item["method"], item.get("status", "unknown"))] += 1
    return [
        {"dataset": dataset, "method": method, "status": status, "n": count}
        for (dataset, method, status), count in sorted(counts.items())
    ]


def _coverage(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in episodes:
        counts[(item["task"]["dataset"], item["method"])][item.get("status", "unknown")] += 1
    result = []
    for (dataset, method), statuses in sorted(counts.items()):
        valid = sum(statuses.get(status, 0) for status in VALID_STATUSES)
        total = sum(statuses.values())
        result.append(
            {
                "dataset": dataset,
                "method": method,
                "valid_n": valid,
                "total_records": total,
                "valid_fraction": valid / total if total else None,
            }
        )
    return result


def _contrast(
    rows: list[dict[str, Any]],
    treatment: str,
    control: str,
    metric: str,
    *,
    reverse: bool = False,
) -> list[dict[str, Any]]:
    indexed = {(row["dataset"], row["method"]): row for row in rows}
    result = []
    for dataset in sorted({row["dataset"] for row in rows}):
        left = indexed.get((dataset, treatment))
        right = indexed.get((dataset, control))
        if not left or not right or left.get(metric) is None or right.get(metric) is None:
            continue
        difference = float(left[metric]) - float(right[metric])
        result.append(
            {
                "dataset": dataset,
                "treatment": treatment,
                "control": control,
                "metric": metric,
                "treatment_n": left.get("n", left.get("paired_n")),
                "control_n": right.get("n", right.get("paired_n")),
                "effect": -difference if reverse else difference,
            }
        )
    return result
