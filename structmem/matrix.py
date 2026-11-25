from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from structmem.graph import cosine_sparse, structural_features
from structmem.memory import DenseEmbeddingRetriever, lexical_vector, role_similarity
from structmem.types import Cell, Experience


@dataclass(frozen=True)
class Pair:
    query_id: str
    source_id: str
    query_dataset: str
    source_dataset: str
    semantic_similarity: float
    structural_similarity: float
    cell: Cell

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "cell": self.cell.value,
        }


def pair_scores(
    query: Experience,
    source: Experience,
    dense_similarity: float,
) -> tuple[float, float]:
    vocabulary_overlap = cosine_sparse(lexical_vector(query.goal), lexical_vector(source.goal))
    semantic = 0.8 * dense_similarity + 0.2 * vocabulary_overlap
    graph_score = cosine_sparse(structural_features(query.graph), structural_features(source.graph))
    roles = role_similarity(query.roles, source.roles)
    return semantic, 0.55 * graph_score + 0.45 * roles


def classify_pair(semantic: float, structural: float, semantic_threshold: float, structural_threshold: float) -> Cell:
    semantic_near = semantic >= semantic_threshold
    structural_near = structural >= structural_threshold
    return {
        (True, True): Cell.NEAR_NEAR,
        (False, True): Cell.FAR_NEAR,
        (True, False): Cell.NEAR_FAR,
        (False, False): Cell.FAR_FAR,
    }[(semantic_near, structural_near)]


def build_matrix(
    queries: list[Experience],
    sources: list[Experience],
    *,
    semantic_threshold: float = 0.25,
    structural_threshold: float = 0.65,
    per_cell: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    semantic_vectors = DenseEmbeddingRetriever().encode(
        [item.goal for item in queries] + [item.goal for item in sources]
    )
    query_vectors = semantic_vectors[: len(queries)]
    source_vectors = semantic_vectors[len(queries) :]
    grouped: defaultdict[Cell, list[Pair]] = defaultdict(list)
    for query_index, query in enumerate(queries):
        for source_index, source in enumerate(sources):
            if query.experience_id == source.experience_id:
                continue
            dense_similarity = float(query_vectors[query_index] @ source_vectors[source_index])
            semantic, structural = pair_scores(query, source, dense_similarity)
            cell = classify_pair(semantic, structural, semantic_threshold, structural_threshold)
            grouped[cell].append(
                Pair(
                    query_id=query.experience_id,
                    source_id=source.experience_id,
                    query_dataset=query.dataset,
                    source_dataset=source.dataset,
                    semantic_similarity=semantic,
                    structural_similarity=structural,
                    cell=cell,
                )
            )
    rng = random.Random(seed)
    candidates_by_query: defaultdict[str, list[Pair]] = defaultdict(list)
    for candidates in grouped.values():
        for pair in candidates:
            candidates_by_query[pair.query_id].append(pair)
    query_order = [item.experience_id for item in queries]
    rng.shuffle(query_order)
    cell_counts: defaultdict[Cell, int] = defaultdict(int)
    selected: list[Pair] = []
    for query_id in query_order:
        candidates = candidates_by_query.get(query_id, [])
        if not candidates:
            continue
        candidates.sort(
            key=lambda pair: (
                cell_counts[pair.cell] >= per_cell,
                cell_counts[pair.cell],
                -abs(pair.structural_similarity - structural_threshold),
                -abs(pair.semantic_similarity - semantic_threshold),
                pair.source_id,
            )
        )
        chosen = candidates[0]
        selected.append(chosen)
        cell_counts[chosen.cell] += 1
    counts = {
        cell.value: {
            "available": len(grouped[cell]),
            "selected": cell_counts[cell],
        }
        for cell in Cell
    }
    by_query: defaultdict[str, list[Pair]] = defaultdict(list)
    for pair in selected:
        by_query[pair.query_id].append(pair)
    tasks = {}
    for query_id, pairs in by_query.items():
        primary = pairs[0]
        automatic_references = [
            pair.source_id
            for pair in pairs
            if pair.structural_similarity >= structural_threshold
        ]
        if not automatic_references:
            automatic_references = [primary.source_id]
        query = next(item for item in queries if item.experience_id == query_id)
        tasks[query.task_id] = {
            "cell": primary.cell.value,
            "reference_id": primary.source_id,
            "automatic_analogous_ids": sorted(set(automatic_references)),
            "reference_kind": "automatic_pseudo_reference",
            "semantic_similarity": primary.semantic_similarity,
            "structural_similarity": primary.structural_similarity,
        }
    return {
        "schema_version": 2,
        "reference_status": "automatic",
        "reference_provenance": {
            "algorithm": "frozen_dense_lexical_role_graph_v1",
            "human_annotation": False,
            "warning": "These are deterministic pseudo-references, not human gold labels.",
        },
        "semantic_threshold": semantic_threshold,
        "semantic_measure": "0.8 * all-MiniLM-L6-v2 cosine + 0.2 * lexical cosine",
        "structural_threshold": structural_threshold,
        "per_cell": per_cell,
        "seed": seed,
        "counts": counts,
        "pairs": [pair.to_dict() for pair in selected],
        "tasks": tasks,
    }
