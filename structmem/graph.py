from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx

from structmem.types import Role


def trace_graph(actions: list[str], roles: list[Role]) -> dict[str, Any]:
    graph = nx.DiGraph()
    last_by_role: dict[Role, int] = {}
    for index, (action, role) in enumerate(zip(actions, roles, strict=True)):
        graph.add_node(index, action=action, role=role.value, kind="action")
        if index:
            graph.add_edge(index - 1, index, kind="temporal")
        if role in last_by_role:
            graph.add_edge(last_by_role[role], index, kind="role_recurrence")
        if role == Role.VERIFICATION:
            previous = next(
                (i for i in range(index - 1, -1, -1) if roles[i] == Role.STATE_CHANGING_ACTION),
                None,
            )
            if previous is not None:
                graph.add_edge(previous, index, kind="effect_check")
        last_by_role[role] = index
    return nx.node_link_data(graph, edges="edges")


def structural_features(graph_data: dict[str, Any]) -> dict[str, float]:
    graph = nx.node_link_graph(graph_data, edges="edges")
    edge_types = Counter(data.get("kind", "unknown") for _, _, data in graph.edges(data=True))
    node_roles = Counter(data.get("role", "unknown") for _, data in graph.nodes(data=True))
    denom_nodes = max(1, graph.number_of_nodes())
    denom_edges = max(1, graph.number_of_edges())
    features = {f"role:{key}": value / denom_nodes for key, value in node_roles.items()}
    features.update({f"edge:{key}": value / denom_edges for key, value in edge_types.items()})
    features["density"] = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    features["length"] = min(graph.number_of_nodes(), 50) / 50
    return features


def cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    lnorm = sum(value * value for value in left.values()) ** 0.5
    rnorm = sum(value * value for value in right.values()) ** 0.5
    return dot / (lnorm * rnorm) if lnorm and rnorm else 0.0

