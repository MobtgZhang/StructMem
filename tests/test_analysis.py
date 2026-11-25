from structmem.analysis import FACTORIAL_METHODS, analyze
from structmem.types import Cell


def episode(
    method,
    task_id,
    success,
    cell=Cell.FAR_NEAR.value,
    *,
    steps=1,
    semantic_score=None,
    source_roles=None,
    status="completed",
):
    diagnostics = {}
    if semantic_score is not None:
        diagnostics["retrieved_semantic_scores"] = {"source": semantic_score}
    if source_roles:
        diagnostics["retrieved_metadata"] = [
            {
                "id": "source",
                "dataset": "source-domain",
                "roles": source_roles,
            }
        ]
    if method == "ours":
        diagnostics.update(query_node_count=4, query_edge_count=3)
    return {
        "task": {
            "dataset": "d",
            "task_id": task_id,
            "metadata": {"cell": cell, "gold_analogous_ids": ["source"]},
        },
        "method": method,
        "status": status,
        "success": success,
        "score": float(success),
        "steps": [{"action": "a", "observation": "o"}] * steps,
        "retrieved_ids": ["source"] if method != "no_memory" else [],
        "transfer_used": method != "no_memory",
        "gate_verdict": "use" if method != "no_memory" else None,
        "diagnostics": diagnostics,
    }


def test_paper_rq1_rq2_and_error_exclusion() -> None:
    roles = ["PreconditionCheck", "StateChangingAction", "Verification"]
    episodes = [
        episode("ours", "a", True, semantic_score=0.1, source_roles=roles),
        episode("embedding_retrieval", "a", False, semantic_score=0.8),
        episode("ours", "broken", False, semantic_score=0.2, status="agent_error"),
    ]
    result = analyze(episodes)
    assert result["valid_episodes"] == 2
    assert result["excluded_errors"] == 1
    assert result["rq1_semantic_distance"]["ready"]
    patterns = result["rq2_role_pattern_transfer"]["patterns"]
    assert patterns[0]["label"] == "PreconditionCheck -> StateChangingAction -> Verification"
    assert patterns[0]["cross_domain_transfer_rate"] == 1.0


def test_later_valid_retry_supersedes_earlier_agent_error() -> None:
    failed_attempt = episode("ours", "retry", False, status="agent_error")
    successful_retry = episode("ours", "retry", True)

    result = analyze([failed_attempt, successful_retry])

    assert result["raw_episode_records"] == 2
    assert result["resolved_episode_records"] == 1
    assert result["valid_episodes"] == 1
    assert result["excluded_errors"] == 0
    assert result["main_results"][0]["success_rate"] == 1.0


def test_rq3_requires_and_computes_complete_factorial() -> None:
    episodes = []
    for method, (roles, structure, binding) in FACTORIAL_METHODS.items():
        success = bool(roles and structure)
        episodes.append(episode(method, "shared", success))
    rq3 = analyze(episodes)["rq3_component_interactions"]
    assert rq3["ready"]
    effects = {row["interaction"]: row["effect_pp"] for row in rq3["interactions"]}
    assert effects["Roles x Structure"] == 100.0
    assert effects["Roles x Binding"] == 0.0


def test_rq3_reports_task_when_all_factorial_cells_error() -> None:
    episodes = [
        episode(method, "broken", False, status="agent_error")
        for method in FACTORIAL_METHODS
    ]
    rq3 = analyze(episodes)["rq3_component_interactions"]
    assert not rq3["ready"]
    assert rq3["missing"] == [
        {"dataset": "d", "incomplete_tasks": 1, "first_task": "broken"}
    ]


def test_rq4_rejects_unverified_audit_and_accepts_50_verified_cases() -> None:
    episodes = [episode("ours", f"f{index}", False) for index in range(50)]
    pending = {
        "annotation_status": "pending",
        "items": [
            {
                "failure_id": f"d:f{index}",
                "category": "other",
                "annotator_note": "Manually reviewed failure trace.",
            }
            for index in range(50)
        ],
    }
    assert not analyze(episodes, pending)["rq4_failure_modes"]["ready"]
    pending["annotation_status"] = "verified"
    rq4 = analyze(episodes, pending)["rq4_failure_modes"]
    assert rq4["ready"]
    other = next(item for item in rq4["categories"] if item["category"] == "other")
    assert other["fraction"] == 1.0


def test_rq4_rejects_duplicate_or_foreign_failure_ids() -> None:
    episodes = [episode("ours", f"f{index}", False) for index in range(50)]
    items = [
        {
            "failure_id": f"d:f{index}",
            "category": "other",
            "annotator_note": "Reviewed.",
        }
        for index in range(50)
    ]
    items[-1]["failure_id"] = "d:f0"
    audit = {"annotation_status": "verified", "items": items}
    rq4 = analyze(episodes, audit)["rq4_failure_modes"]
    assert not rq4["ready"]
    assert rq4["duplicate_failure_ids"] == ["d:f0"]


def test_rq4_accepts_provenanced_automatic_audit() -> None:
    episodes = [episode("ours", f"f{index}", False) for index in range(50)]
    audit = {
        "annotation_status": "automatic_complete",
        "provenance": {
            "classifier": "llm",
            "model": "judge-model",
            "prompt_version": "rq4_failure_judge_v1",
            "human_annotation": False,
        },
        "items": [
            {
                "failure_id": f"d:f{index}",
                "category": "other",
                "confidence": 0.75,
                "explanation": "The trace does not support a more specific category.",
            }
            for index in range(50)
        ],
    }
    rq4 = analyze(episodes, audit)["rq4_failure_modes"]
    assert rq4["ready"]
    assert rq4["automatic_complete"]
    assert not rq4["manual_verified"]
