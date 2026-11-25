from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from structmem.llm import LLMClient
from structmem.prompts import action_messages, extract_action
from structmem.types import Episode, Experience, TaskSpec


@dataclass
class AgentContext:
    task: TaskSpec
    history: list[tuple[str, str]] = field(default_factory=list)
    retrieved: list[Experience] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    model_usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def record_usage(self, response: Any) -> None:
        self.model_usage["input_tokens"] += int(response.input_tokens)
        self.model_usage["output_tokens"] += int(response.output_tokens)


class Agent:
    name = "agent"

    def __init__(
        self,
        llm: LLMClient,
        *,
        token_budget: int = 2000,
        top_k: int = 5,
        prefilter_k: int = 50,
        allow_online_updates: bool = False,
    ):
        self.llm = llm
        self.token_budget = token_budget
        self.top_k = top_k
        self.prefilter_k = prefilter_k
        self.allow_online_updates = allow_online_updates

    def begin(self, task: TaskSpec, query: Experience, memory: list[Experience]) -> AgentContext:
        return AgentContext(task=task)

    def guidance(self, context: AgentContext, observation: str) -> str:
        return ""

    def act(
        self,
        context: AgentContext,
        observation: str,
        admissible_actions: list[str] | None,
    ) -> tuple[str, dict[str, int]]:
        response = self.llm.complete(
            action_messages(
                context.task.goal,
                observation,
                context.history,
                self.guidance(context, observation),
                admissible_actions,
            ),
            max_tokens=512 if context.task.dataset == "agentbench" else 128,
        )
        action = extract_action(response.text, admissible_actions)
        context.history.append((observation, action))
        return action, {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}

    def end(self, context: AgentContext, episode: Episode) -> None:
        pass


def fit_budget(parts: list[str], token_budget: int) -> str:
    character_budget = token_budget * 4
    selected: list[str] = []
    used = 0
    for part in parts:
        if used + len(part) > character_budget:
            remaining = character_budget - used
            if remaining > 80:
                selected.append(part[:remaining])
            break
        selected.append(part)
        used += len(part)
    return "\n".join(selected)
