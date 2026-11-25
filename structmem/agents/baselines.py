from __future__ import annotations

from collections import defaultdict

from structmem.agents.base import Agent, AgentContext, fit_budget
from structmem.memory import (
    BM25Retriever,
    DenseEmbeddingRetriever,
    GraphEmbeddingRetriever,
    HybridRetriever,
    SemanticRetriever,
    diversified,
)
from structmem.prompts import compact_experience
from structmem.types import Episode, Experience, TaskSpec


class NoMemory(Agent):
    name = "no_memory"


class RetrievalBaseline(Agent):
    label = "Retrieved experiences:"

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        ranked = self.retriever.rank(query, memory)
        context = AgentContext(task=task, retrieved=[item for item, _ in ranked[: self.top_k]])
        context.diagnostics["ranked_ids"] = [item.experience_id for item, _ in ranked]
        context.diagnostics["ranked_scores"] = {item.experience_id: score for item, score in ranked}
        return context

    def guidance(self, context: AgentContext, observation: str) -> str:
        blocks = [self.label, *(compact_experience(item) for item in context.retrieved)]
        return fit_budget(blocks, self.token_budget)


class BM25(RetrievalBaseline):
    name = "bm25"
    label = "BM25 lexical matches; adapt them to the current state:"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = BM25Retriever()


class EmbeddingRetrieval(RetrievalBaseline):
    name = "embedding_retrieval"
    label = "Dense semantic matches; adapt them to the current state:"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = DenseEmbeddingRetriever()

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        pool = [item for item in memory if item.dataset == task.dataset and item.success]
        context = super().begin(task, query, pool)
        context.diagnostics["candidate_pool_policy"] = "same_dataset_success_only"
        context.diagnostics["candidate_pool_size"] = len(pool)
        context.diagnostics["retrieved_semantic_scores"] = {
            item.experience_id: context.diagnostics["ranked_scores"][item.experience_id]
            for item in context.retrieved
        }
        return context


class GraphEmbedding(RetrievalBaseline):
    name = "graph_embedding"
    label = "Graph-embedding matches; adapt roles and actions to the current state:"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = GraphEmbeddingRetriever()

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        payload, response = self.llm.json(
            [
                {
                    "role": "system",
                    "content": (
                        "Represent the task as an anticipated execution graph. Return JSON with nodes "
                        "[{action,role}] and edges [{source,target,type}]. Roles must be one of "
                        "InformationAcquisition, PreconditionCheck, StateChangingAction, Verification, "
                        "Recovery, Termination."
                    ),
                },
                {"role": "user", "content": f"Task: {query.goal}"},
            ],
            max_tokens=500,
        )
        query = _apply_graph_payload(query, payload)
        context = super().begin(task, query, memory)
        context.record_usage(response)
        return context


class Voyager(Agent):
    """Growing executable skill library with semantic skill retrieval."""

    name = "voyager"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skills: list[Experience] = []
        self.retriever = SemanticRetriever()

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        available = self.skills or [item for item in memory if item.success]
        selected = [item for item, _ in self.retriever.rank(query, available)[: self.top_k]]
        context = AgentContext(task=task, retrieved=selected)
        context.diagnostics["library_size"] = len(available)
        return context

    def guidance(self, context: AgentContext, observation: str) -> str:
        blocks = ["Reusable skills; adapt object names and arguments to the current environment."]
        blocks.extend(compact_experience(item) for item in context.retrieved)
        return fit_budget(blocks, self.token_budget)

    def end(self, context: AgentContext, episode: Episode) -> None:
        if self.allow_online_updates and episode.success:
            experience = episode_to_experience(episode, lesson="Successful executable skill")
            self.skills.append(experience)


class Reflexion(Agent):
    """Episodic verbal feedback generated after failed trials."""

    name = "reflexion"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflections: list[str] = []
        self.retriever = SemanticRetriever()

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        failures = [item for item in memory if not item.success]
        selected = [item for item, _ in self.retriever.rank(query, failures)[: self.top_k]]
        return AgentContext(task=task, retrieved=selected)

    def guidance(self, context: AgentContext, observation: str) -> str:
        traces = [
            f"Training failure to reflect on: goal={item.goal}; actions={item.actions}; lesson={item.lesson}"
            for item in context.retrieved
        ]
        return fit_budget(["Failure reflections:", *traces, *self.reflections[-8:]], self.token_budget)

    def end(self, context: AgentContext, episode: Episode) -> None:
        if not self.allow_online_updates or episode.success or not episode.steps:
            return
        trace = "\n".join(f"{step.action} -> {step.observation[:180]}" for step in episode.steps[-10:])
        response = self.llm.complete(
            [
                {"role": "system", "content": "Diagnose an interactive-agent failure. State one concrete cause and one actionable correction. Do not invent environment facts."},
                {"role": "user", "content": f"Task: {episode.task.goal}\nTrace:\n{trace}\nReflection:"},
            ],
            max_tokens=180,
        )
        self.reflections.append(response.text.strip())


class ExpeL(Agent):
    """Contrastive rules induced from successes and failures."""

    name = "expel"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules: list[str] = []
        self.recent: list[Episode] = []
        self.retriever = SemanticRetriever()

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        ranked = self.retriever.rank(query, memory)
        selected = [item for item, _ in ranked[: max(2, self.top_k)]]
        context = AgentContext(task=task, retrieved=selected)
        successes = [item for item in selected if item.success]
        failures = [item for item in selected if not item.success]
        if successes and failures:
            response = self.llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Infer up to three concise, reusable action-selection rules by contrasting the supplied "
                            "successful and failed training experiences. Do not mention source object names."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Successes: {[(item.goal, item.actions) for item in successes]}\n"
                            f"Failures: {[(item.goal, item.actions) for item in failures]}\nRules:"
                        ),
                    },
                ],
                max_tokens=240,
            )
            context.record_usage(response)
            context.diagnostics["episode_rules"] = response.text.strip()
        return context

    def guidance(self, context: AgentContext, observation: str) -> str:
        return fit_budget(
            ["Learned cross-task rules:", context.diagnostics.get("episode_rules", ""), *self.rules[-12:]],
            self.token_budget,
        )

    def end(self, context: AgentContext, episode: Episode) -> None:
        if not self.allow_online_updates:
            return
        self.recent.append(episode)
        if len(self.recent) < 2:
            return
        success = next((item for item in reversed(self.recent) if item.success), None)
        failure = next((item for item in reversed(self.recent) if not item.success), None)
        if not success or not failure:
            return
        response = self.llm.complete(
            [
                {"role": "system", "content": "Induce one general, concise action-selection rule from a successful and failed trajectory. Avoid task-specific object names."},
                {
                    "role": "user",
                    "content": f"Successful actions: {[s.action for s in success.steps]}\nFailed actions: {[s.action for s in failure.steps]}\nRule:",
                },
            ],
            max_tokens=120,
        )
        rule = response.text.strip()
        if rule and rule not in self.rules:
            self.rules.append(rule)


class MemGPT(Agent):
    """Bounded working context plus archival memory recall and consolidation."""

    name = "memgpt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archival: list[Experience] = []
        self.retriever = SemanticRetriever()

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        archive = self.archival or memory
        selected = [item for item, _ in self.retriever.rank(query, archive)[: self.top_k]]
        return AgentContext(task=task, retrieved=selected)

    def guidance(self, context: AgentContext, observation: str) -> str:
        blocks = ["Archival recall:", *(compact_experience(item) for item in context.retrieved)]
        return fit_budget(blocks, self.token_budget)

    def end(self, context: AgentContext, episode: Episode) -> None:
        if self.allow_online_updates and episode.success:
            self.archival.append(episode_to_experience(episode, lesson="Archived successful episode"))


class MCMA(Agent):
    """Aligns task text, action traces, and outcome memories before selection."""

    name = "mcma"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = HybridRetriever()

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        ranked = self.retriever.rank(query, memory)
        selected = diversified(ranked, self.top_k)
        return AgentContext(task=task, retrieved=selected)

    def guidance(self, context: AgentContext, observation: str) -> str:
        blocks = ["Aligned multi-source memories (goal, roles, actions, outcome):"]
        blocks.extend(compact_experience(item) for item in context.retrieved)
        return fit_budget(blocks, self.token_budget)


class AgentKB(Agent):
    """Knowledge-base retrieval with explicit relevance and reliability gates."""

    name = "agentkb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = BM25Retriever()
        self.usage: defaultdict[str, list[bool]] = defaultdict(list)

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        ranked = self.retriever.rank(query, memory)
        gated = []
        for item, score in ranked:
            history = self.usage[item.experience_id]
            reliability = sum(history) / len(history) if history else float(item.success)
            if score > 0 and reliability >= 0.5:
                gated.append(item)
            if len(gated) == self.top_k:
                break
        context = AgentContext(task=task, retrieved=gated)
        context.diagnostics["gate_candidates"] = len(ranked)
        return context

    def guidance(self, context: AgentContext, observation: str) -> str:
        blocks = ["Knowledge-base entries that passed relevance and reliability gates:"]
        blocks.extend(compact_experience(item) for item in context.retrieved)
        return fit_budget(blocks, self.token_budget)

    def end(self, context: AgentContext, episode: Episode) -> None:
        if not self.allow_online_updates:
            return
        for item in context.retrieved:
            self.usage[item.experience_id].append(episode.success)


def episode_to_experience(episode: Episode, lesson: str) -> Experience:
    from structmem.graph import trace_graph
    from structmem.roles import annotate_trace

    actions = [step.action for step in episode.steps]
    observations = [step.observation for step in episode.steps]
    roles = annotate_trace(actions, observations)
    return Experience(
        experience_id=f"{episode.task.dataset}:{episode.task.task_id}:{episode.method}",
        dataset=episode.task.dataset,
        task_id=episode.task.task_id,
        goal=episode.task.goal,
        success=episode.success,
        score=episode.score,
        actions=actions,
        observations=observations,
        roles=roles,
        graph=trace_graph(actions, roles),
        lesson=lesson,
    )


def _apply_graph_payload(query: Experience, payload: dict) -> Experience:
    from structmem.graph import trace_graph
    from structmem.types import Role

    actions = []
    roles = []
    for node in payload.get("nodes", []):
        try:
            action = str(node["action"]).strip()
            role = Role(node["role"])
        except (KeyError, TypeError, ValueError):
            continue
        if action:
            actions.append(action)
            roles.append(role)
    if not actions:
        return query
    query.actions = actions
    query.observations = [query.goal] * len(actions)
    query.roles = roles
    query.graph = trace_graph(actions, roles)
    edges = []
    for edge in payload.get("edges", []):
        try:
            source, target = int(edge["source"]), int(edge["target"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= source < len(actions) and 0 <= target < len(actions) and source != target:
            edges.append({"source": source, "target": target, "kind": str(edge.get("type", "causal"))})
    if edges:
        query.graph["edges"] = edges
    return query
