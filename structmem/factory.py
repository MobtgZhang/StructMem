from __future__ import annotations

from structmem.agents import (
    AgentKB, BM25, EmbeddingRetrieval, ExpeL, GraphEmbedding, MCMA, MemGPT,
    NoMemory, Reflexion, StructuralAgent, Voyager,
)
from structmem.agents.base import Agent
from structmem.llm import LLMClient


METHODS = (
    "no_memory",
    "bm25",
    "embedding_retrieval",
    "graph_embedding",
    "voyager",
    "reflexion",
    "expel",
    "memgpt",
    "mcma",
    "agentkb",
    "ours",
    "w_o_roles",
    "w_o_structure",
    "w_o_binding",
    "w_o_gate",
    "w_o_roles_structure",
    "w_o_roles_binding",
    "w_o_structure_binding",
    "w_o_roles_structure_binding",
)


def make_agent(
    name: str,
    llm: LLMClient,
    *,
    token_budget: int,
    top_k: int,
    prefilter_k: int = 50,
    allow_online_updates: bool = False,
) -> Agent:
    common = {
        "llm": llm,
        "token_budget": token_budget,
        "top_k": top_k,
        "prefilter_k": prefilter_k,
        "allow_online_updates": allow_online_updates,
    }
    factories = {
        "no_memory": lambda: NoMemory(**common),
        "bm25": lambda: BM25(**common),
        "embedding_retrieval": lambda: EmbeddingRetrieval(**common),
        "graph_embedding": lambda: GraphEmbedding(**common),
        "voyager": lambda: Voyager(**common),
        "reflexion": lambda: Reflexion(**common),
        "expel": lambda: ExpeL(**common),
        "memgpt": lambda: MemGPT(**common),
        "mcma": lambda: MCMA(**common),
        "agentkb": lambda: AgentKB(**common),
        "ours": lambda: StructuralAgent(**common),
        "w_o_roles": lambda: StructuralAgent(**common, use_roles=False),
        "w_o_structure": lambda: StructuralAgent(**common, use_structure=False),
        "w_o_binding": lambda: StructuralAgent(**common, use_binding=False),
        "w_o_gate": lambda: StructuralAgent(**common, conservative_gate=False),
        "w_o_roles_structure": lambda: StructuralAgent(
            **common, use_roles=False, use_structure=False
        ),
        "w_o_roles_binding": lambda: StructuralAgent(
            **common, use_roles=False, use_binding=False
        ),
        "w_o_structure_binding": lambda: StructuralAgent(
            **common, use_structure=False, use_binding=False
        ),
        "w_o_roles_structure_binding": lambda: StructuralAgent(
            **common, use_roles=False, use_structure=False, use_binding=False
        ),
    }
    try:
        agent = factories[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown method {name!r}; expected one of {METHODS}") from exc
    agent.name = name
    return agent
