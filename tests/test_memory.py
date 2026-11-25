import json
from types import SimpleNamespace

from structmem.graph import trace_graph
from structmem.llm import LLMResponse
from structmem.memory import BM25Retriever, StructureRetriever, role_similarity
from structmem.matrix import build_matrix
from structmem.prepare import failure_audit_from_episodes
from structmem.types import Experience, Role


def experience(identifier: str, goal: str, roles: list[Role]) -> Experience:
    actions = [f"action-{index}" for index in range(len(roles))]
    return Experience(
        experience_id=identifier,
        dataset="test",
        task_id=identifier,
        goal=goal,
        success=True,
        score=1.0,
        actions=actions,
        observations=[""] * len(actions),
        roles=roles,
        graph=trace_graph(actions, roles),
    )


def test_structural_retrieval_ignores_surface_domain() -> None:
    pattern = [Role.PRECONDITION_CHECK, Role.STATE_CHANGING_ACTION, Role.VERIFICATION]
    query = experience("q", "heat the mug and verify it", pattern)
    analogous = experience("analogous", "activate the reactor and measure temperature", pattern)
    lexical = experience(
        "lexical",
        "find the mug beside another mug",
        [Role.INFORMATION_ACQUISITION, Role.INFORMATION_ACQUISITION],
    )
    ranked = StructureRetriever().rank(query, [lexical, analogous])
    assert ranked[0][0].experience_id == "analogous"


def test_bm25_prefers_shared_terms() -> None:
    query = experience("q", "clean mug cabinet", [Role.STATE_CHANGING_ACTION])
    near = experience("near", "put clean mug in cabinet", [Role.INFORMATION_ACQUISITION])
    far = experience("far", "measure acid temperature", [Role.STATE_CHANGING_ACTION])
    assert BM25Retriever().rank(query, [far, near])[0][0].experience_id == "near"


def test_role_lcs_is_normalized() -> None:
    assert role_similarity([Role.VERIFICATION], [Role.VERIFICATION]) == 1.0
    assert role_similarity([Role.VERIFICATION], [Role.RECOVERY]) == 0.0


def test_matrix_assigns_each_query_to_at_most_one_cell(monkeypatch) -> None:
    pattern = [Role.PRECONDITION_CHECK, Role.STATE_CHANGING_ACTION, Role.VERIFICATION]
    queries = [experience(f"q{index}", f"query {index}", pattern) for index in range(4)]
    sources = [experience("source", "source task", pattern)]
    monkeypatch.setattr(
        "structmem.matrix.DenseEmbeddingRetriever.encode",
        lambda self, texts: __import__("numpy").ones((len(texts), 2)) / (2**0.5),
    )
    result = build_matrix(queries, sources, semantic_threshold=0.0, structural_threshold=0.0, per_cell=4)
    query_ids = [pair["query_id"] for pair in result["pairs"]]
    assert len(query_ids) == len(set(query_ids))
    assert len(query_ids) == len(queries)
    assert result["reference_status"] == "automatic"
    assert all(
        task["automatic_analogous_ids"] for task in result["tasks"].values()
    )


def test_automatic_failure_audit_records_provenance(tmp_path) -> None:
    class FakeLLM:
        config = SimpleNamespace(model="judge-model")

        def json(self, messages, *, max_tokens):
            return (
                {
                    "category": "hidden_preconditions",
                    "confidence": 0.8,
                    "explanation": "The selected action was blocked by an unseen state constraint.",
                },
                LLMResponse(text="{}", input_tokens=20, output_tokens=8),
            )

    episode_path = tmp_path / "episodes.jsonl"
    episode_path.write_text(
        json.dumps(
            {
                "task": {"dataset": "d", "task_id": "f0", "goal": "finish task"},
                "method": "ours",
                "status": "task_failed",
                "success": False,
                "score": 0.0,
                "steps": [{"action": "act", "observation": "blocked"}],
                "retrieved_ids": ["source"],
                "transfer_used": True,
                "binding_outcome": "clean",
                "gate_verdict": "use",
                "diagnostics": {"ranked_scores": {"source": 0.9}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    result = failure_audit_from_episodes(
        [episode_path], output, llm=FakeLLM(), limit=1
    )
    assert result["annotation_status"] == "automatic_complete"
    assert result["provenance"]["human_annotation"] is False
    assert result["items"][0]["category"] == "hidden_preconditions"
    assert result["token_usage"] == {"input_tokens": 20, "output_tokens": 8}
