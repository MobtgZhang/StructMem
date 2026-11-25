from __future__ import annotations

import json
import math
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from structmem.graph import cosine_sparse, structural_features
from structmem.types import Experience, Role


TOKEN_RE = re.compile(r"[a-z0-9_]+", re.I)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def lexical_vector(text: str) -> dict[str, float]:
    counts = Counter(tokens(text))
    norm = sum(value * value for value in counts.values()) ** 0.5
    return {key: value / norm for key, value in counts.items()} if norm else {}


def role_similarity(left: list[Role], right: list[Role]) -> float:
    if not left or not right:
        return 0.0
    rows, cols = len(left) + 1, len(right) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            matrix[i][j] = (
                matrix[i - 1][j - 1] + 1
                if left[i - 1] == right[j - 1]
                else max(matrix[i - 1][j], matrix[i][j - 1])
            )
    return matrix[-1][-1] / max(len(left), len(right))


class ExperienceStore:
    def __init__(self, experiences: Iterable[Experience] = ()):
        self._items: dict[str, Experience] = {item.experience_id: item for item in experiences}

    def add(self, experience: Experience) -> None:
        self._items[experience.experience_id] = experience

    def values(self) -> list[Experience]:
        return list(self._items.values())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps([item.to_dict() for item in self.values()], ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ExperienceStore":
        if not path.exists():
            return cls()
        return cls(Experience.from_dict(item) for item in json.loads(path.read_text(encoding="utf-8")))


class Retriever:
    def rank(self, query: Experience, candidates: list[Experience]) -> list[tuple[Experience, float]]:
        raise NotImplementedError


class BM25Retriever(Retriever):
    def rank(self, query: Experience, candidates: list[Experience]) -> list[tuple[Experience, float]]:
        if not candidates:
            return []
        documents = [tokens(item.goal + " " + item.lesson) for item in candidates]
        average_length = sum(map(len, documents)) / len(documents)
        frequencies = Counter(token for document in map(set, documents) for token in document)
        query_terms = set(tokens(query.goal))
        ranked: list[tuple[Experience, float]] = []
        for item, document in zip(candidates, documents, strict=True):
            counts = Counter(document)
            score = 0.0
            for term in query_terms:
                document_frequency = frequencies.get(term, 0)
                inverse = math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
                frequency = counts.get(term, 0)
                denom = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / max(1, average_length))
                score += inverse * frequency * 2.5 / denom if denom else 0.0
            ranked.append((item, score))
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0].experience_id))


class SemanticRetriever(Retriever):
    """Deterministic lexical-semantic fallback used without a separate embedding service."""

    def rank(self, query: Experience, candidates: list[Experience]) -> list[tuple[Experience, float]]:
        query_vector = lexical_vector(query.goal)
        return sorted(
            ((item, cosine_sparse(query_vector, lexical_vector(item.goal + " " + item.lesson))) for item in candidates),
            key=lambda pair: (-pair[1], pair[0].experience_id),
        )


class DenseEmbeddingRetriever(Retriever):
    """SentenceTransformer cosine retrieval with an explicitly local model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        return self._model

    def rank(self, query: Experience, candidates: list[Experience]) -> list[tuple[Experience, float]]:
        if not candidates:
            return []
        texts = [query.goal, *(item.goal + " " + item.lesson for item in candidates)]
        vectors = self.encode(texts)
        query_vector = vectors[0]
        ranked = [
            (item, float(query_vector @ vector))
            for item, vector in zip(candidates, vectors[1:], strict=True)
        ]
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0].experience_id))

    def encode(self, texts: list[str]):
        return self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class GraphEmbeddingRetriever(Retriever):
    """Two-round WL message-passing graph embeddings with cosine retrieval."""

    def __init__(self, dimensions: int = 256, rounds: int = 2):
        self.dimensions = dimensions
        self.rounds = rounds

    def rank(self, query: Experience, candidates: list[Experience]) -> list[tuple[Experience, float]]:
        query_vector = self._embed(query.graph)
        ranked = [(item, cosine_sparse(query_vector, self._embed(item.graph))) for item in candidates]
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0].experience_id))

    def _embed(self, graph_data: dict) -> dict[str, float]:
        import networkx as nx

        graph = nx.node_link_graph(graph_data, edges="edges")
        labels = {
            node: str(data.get("role", data.get("kind", "unknown")))
            for node, data in graph.nodes(data=True)
        }
        features: Counter[str] = Counter(labels.values())
        for _ in range(self.rounds):
            updated = {}
            for node in graph.nodes:
                neighborhood = []
                for source, _, data in graph.in_edges(node, data=True):
                    neighborhood.append(f"in:{data.get('kind', 'edge')}:{labels[source]}")
                for _, target, data in graph.out_edges(node, data=True):
                    neighborhood.append(f"out:{data.get('kind', 'edge')}:{labels[target]}")
                signature = labels[node] + "|" + "|".join(sorted(neighborhood))
                updated[node] = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
            labels = updated
            features.update(labels.values())
        projected: defaultdict[str, float] = defaultdict(float)
        for feature, value in features.items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            projected[str(index)] += sign * value
        return dict(projected)


class StructureRetriever(Retriever):
    def __init__(self, *, use_roles: bool = True, use_edges: bool = True):
        self.use_roles = use_roles
        self.use_edges = use_edges

    def rank(self, query: Experience, candidates: list[Experience]) -> list[tuple[Experience, float]]:
        query_features = structural_features(query.graph)
        if not self.use_roles:
            query_features = {key: value for key, value in query_features.items() if not key.startswith("role:")}
        if not self.use_edges:
            query_features = {key: value for key, value in query_features.items() if not key.startswith("edge:")}
        ranked = []
        for item in candidates:
            features = structural_features(item.graph)
            if not self.use_roles:
                features = {key: value for key, value in features.items() if not key.startswith("role:")}
            if not self.use_edges:
                features = {key: value for key, value in features.items() if not key.startswith("edge:")}
            feature_score = cosine_sparse(query_features, features)
            sequence_score = role_similarity(query.roles, item.roles) if self.use_roles else 0.0
            score = 0.55 * feature_score + 0.45 * sequence_score if self.use_roles else feature_score
            ranked.append((item, score))
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0].experience_id))


class HybridRetriever(Retriever):
    def __init__(self):
        self.semantic = SemanticRetriever()
        self.structural = StructureRetriever()

    def rank(self, query: Experience, candidates: list[Experience]) -> list[tuple[Experience, float]]:
        semantic = {item.experience_id: score for item, score in self.semantic.rank(query, candidates)}
        structural = {item.experience_id: score for item, score in self.structural.rank(query, candidates)}
        return sorted(
            ((item, 0.35 * semantic[item.experience_id] + 0.65 * structural[item.experience_id]) for item in candidates),
            key=lambda pair: (-pair[1], pair[0].experience_id),
        )


def diversified(ranked: list[tuple[Experience, float]], limit: int) -> list[Experience]:
    result: list[Experience] = []
    domain_counts: defaultdict[str, int] = defaultdict(int)
    for experience, _ in ranked:
        if domain_counts[experience.dataset] >= max(1, limit // 2):
            continue
        result.append(experience)
        domain_counts[experience.dataset] += 1
        if len(result) == limit:
            return result
    for experience, _ in ranked:
        if experience not in result:
            result.append(experience)
        if len(result) == limit:
            break
    return result
