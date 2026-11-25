from __future__ import annotations

import time
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from structmem.agents.baselines import episode_to_experience
from structmem.config import ExperimentConfig
from structmem.environments import make_environment
from structmem.environments.base import InfrastructureError
from structmem.factory import make_agent
from structmem.graph import trace_graph
from structmem.llm import LLMClient, LLMError
from structmem.memory import ExperienceStore
from structmem.roles import annotate_trace
from structmem.storage import JSONLStore, completed_keys, load_completed_keys, manifest, write_json
from structmem.types import Episode, Experience, RunStatus, Step, TaskSpec


def empty_query(task: TaskSpec) -> Experience:
    roles = annotate_trace([task.goal], [task.goal])
    return Experience(
        experience_id=f"query:{task.dataset}:{task.task_id}",
        dataset=task.dataset,
        task_id=task.task_id,
        goal=task.goal,
        success=False,
        score=0.0,
        actions=[task.goal],
        observations=[task.goal],
        roles=roles,
        graph=trace_graph([task.goal], roles),
    )


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.output = config.output_dir
        self.output.mkdir(parents=True, exist_ok=True)
        self.store = JSONLStore(self.output / "episodes.jsonl")
        self.memory = ExperienceStore.load(config.memory_file) if config.memory_file else ExperienceStore()
        self.llm = LLMClient(config.api)

    def run(self) -> None:
        write_json(self.output / "manifest.json", manifest(self.config_path, self.config))
        existing = self.store.read() if self.config.resume else []
        done_keys = completed_keys(existing)
        if self.config.completed_keys_file:
            done_keys.update(load_completed_keys(self.config.completed_keys_file))
        for dataset_name, dataset_config in self.config.datasets.items():
            if not dataset_config.enabled:
                continue
            for method in self.config.methods:
                environment = None
                try:
                    environment = make_environment(
                        dataset_name,
                        dataset_config.split,
                        {**dataset_config.options, "task_limit": dataset_config.tasks},
                        self.config.seed,
                    )
                    tasks = environment.tasks(dataset_config.tasks)
                    tasks = self._attach_pair_manifest(tasks, dataset_config.options.get("pair_manifest"))
                    agent = make_agent(
                        method,
                        self.llm,
                        token_budget=self.config.memory_token_budget,
                        top_k=self.config.top_k,
                        prefilter_k=self.config.prefilter_k,
                        allow_online_updates=self.config.online_memory_updates,
                    )
                    for task in tasks:
                        key = (dataset_name, method, task.task_id)
                        if key in done_keys:
                            environment.advance(task)
                            continue
                        episode = self._episode(environment, agent, task, dataset_config.max_steps)
                        self.store.append(episode.to_dict())
                        if hasattr(agent, "_last_context"):
                            agent.end(agent._last_context, episode)
                        if self.config.online_memory_updates and episode.success:
                            self.memory.add(episode_to_experience(episode, lesson="Online successful evaluation episode"))
                except InfrastructureError as exc:
                    self._record_infrastructure_error(dataset_name, method, dataset_config.split, exc)
                finally:
                    if environment:
                        environment.close()
        if self.config.online_memory_updates:
            self.memory.save(self.output / "memory_after_run.json")

    def _episode(self, environment: Any, agent: Any, task: TaskSpec, max_steps: int) -> Episode:
        started = time.monotonic()
        steps: list[Step] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        context = None
        try:
            reset = environment.reset(task)
            actual_goal = str(reset.info.get("goal") or task.goal)
            task = replace(task, goal=actual_goal)
            query = empty_query(task)
            context = agent.begin(task, query, self.memory.values())
            context.diagnostics["retrieved_metadata"] = [
                {
                    "id": item.experience_id,
                    "dataset": item.dataset,
                    "task_id": item.task_id,
                    "goal": item.goal,
                    "roles": [role.value for role in item.roles],
                }
                for item in context.retrieved
            ]
            agent._last_context = context
            observation = reset.observation
            admissible = reset.admissible_actions
            final_success = False
            final_score = 0.0
            done = False
            for _ in range(max_steps):
                action, action_usage = agent.act(context, observation, admissible)
                for key in usage:
                    usage[key] += action_usage[key]
                outcome = environment.step(action)
                steps.append(
                    Step(
                        observation=outcome.observation,
                        action=action,
                        reward=outcome.reward,
                        done=outcome.done,
                        info=outcome.info,
                    )
                )
                observation = outcome.observation
                admissible = outcome.admissible_actions
                final_success = outcome.success
                final_score = outcome.score
                done = outcome.done
                if done:
                    break
            status = RunStatus.COMPLETED if final_success else RunStatus.TASK_FAILED
            diagnostics = context.diagnostics
            for key in usage:
                usage[key] += context.model_usage[key]
            return Episode(
                task=task,
                method=agent.name,
                status=status,
                success=final_success,
                score=final_score,
                steps=steps,
                retrieved_ids=[item.experience_id for item in context.retrieved],
                transfer_used=(
                    diagnostics.get("gate_verdict") in {"use", "probe"}
                    if diagnostics.get("gate_verdict") is not None
                    else bool(context.retrieved)
                ),
                binding_outcome=diagnostics.get("binding_outcome"),
                gate_verdict=diagnostics.get("gate_verdict"),
                token_usage=usage,
                latency_seconds=time.monotonic() - started,
                diagnostics=diagnostics,
            )
        except LLMError as exc:
            if context is not None:
                for key in usage:
                    usage[key] += context.model_usage[key]
            return Episode(
                task=task,
                method=agent.name,
                status=RunStatus.AGENT_ERROR,
                success=False,
                score=0.0,
                steps=steps,
                error=str(exc),
                token_usage=usage,
                latency_seconds=time.monotonic() - started,
            )
        except InfrastructureError:
            raise
        except Exception as exc:
            if context is not None:
                for key in usage:
                    usage[key] += context.model_usage[key]
            return Episode(
                task=task,
                method=agent.name,
                status=RunStatus.AGENT_ERROR,
                success=False,
                score=0.0,
                steps=steps,
                error=f"{type(exc).__name__}: {exc}",
                token_usage=usage,
                latency_seconds=time.monotonic() - started,
            )

    def _record_infrastructure_error(self, dataset: str, method: str, split: str, exc: Exception) -> None:
        task = TaskSpec(dataset=dataset, task_id="__infrastructure__", goal="", split=split, native_id=None)
        episode = Episode(
            task=task,
            method=method,
            status=RunStatus.INFRASTRUCTURE_ERROR,
            success=False,
            score=0.0,
            steps=[],
            error=f"{type(exc).__name__}: {exc}",
        )
        self.store.append(episode.to_dict())

    def _attach_pair_manifest(self, tasks: list[TaskSpec], path_value: str | None) -> list[TaskSpec]:
        if not path_value:
            return tasks
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = (self.config_path.parent / path).resolve()
        if not path.exists():
            raise InfrastructureError(f"Pair manifest not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        status = raw.get("reference_status", raw.get("annotation_status"))
        if status not in {"automatic", "verified"}:
            raise InfrastructureError(
                f"Pair manifest {path} must declare reference_status=automatic or annotation_status=verified"
            )
        if status == "automatic" and not raw.get("reference_provenance", {}).get("algorithm"):
            raise InfrastructureError(f"Automatic pair manifest {path} lacks algorithm provenance")
        mapping = raw.get("tasks", raw)
        attached = []
        missing = []
        empty_references = []
        for task in tasks:
            pair = mapping.get(task.task_id)
            if pair is None:
                missing.append(task.task_id)
                continue
            references = (
                pair.get("automatic_analogous_ids")
                if status == "automatic"
                else pair.get("gold_analogous_ids")
            )
            if not references:
                empty_references.append(task.task_id)
                continue
            attached.append(
                replace(
                    task,
                    metadata={
                        **task.metadata,
                        **pair,
                        "analogous_reference_ids": references,
                        "reference_status": status,
                        "reference_provenance": raw.get("reference_provenance", {}),
                    },
                )
            )
        if missing:
            raise InfrastructureError(
                f"Pair manifest {path} lacks {len(missing)} requested tasks; first missing id: {missing[0]}"
            )
        if empty_references:
            raise InfrastructureError(
                f"Pair manifest {path} has {len(empty_references)} tasks without analogous "
                f"references; first empty task: {empty_references[0]}"
            )
        return attached
