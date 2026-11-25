from types import SimpleNamespace

import pytest

from structmem.environments import agentbench as agentbench_module
from structmem.environments.agentbench import AgentBenchEnvironment
from structmem.environments.base import InfrastructureError
from structmem.environments.compat import patch_textworld_eval
from structmem.environments.scienceworld import ScienceWorldEnvironment
from structmem.types import TaskSpec


class Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = "response"

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_agentbench_protocol_and_result(monkeypatch) -> None:
    posts = []

    monkeypatch.setattr(agentbench_module.requests, "get", lambda *args, **kwargs: Response(list(range(300))))

    def post(url, json, **kwargs):
        posts.append((url, json))
        if url.endswith("start_sample"):
            return Response(
                {
                    "session_id": 7,
                    "output": {
                        "status": "running",
                        "history": [{"role": "user", "content": "Question: count rows"}],
                    },
                }
            )
        return Response(
            {
                "output": {
                    "status": "completed",
                    "result": {"is_correct": True},
                    "history": [{"role": "user", "content": "done"}],
                }
            }
        )

    monkeypatch.setattr(agentbench_module.requests, "post", post)
    env = AgentBenchEnvironment(
        "standard", {"suite": "dbbench-std", "protocol": "v02_text"}, 42
    )
    task = env.tasks(300)[0]
    reset = env.reset(task)
    outcome = env.step('{"tool":"commit_final_answer","answers":["1"]}')
    assert "count rows" in reset.observation
    assert posts[-1][1]["agent_response"]["status"] == "normal"
    assert outcome.done and outcome.success and outcome.score == 1.0


def test_textworld_eval_uses_explicit_namespace() -> None:
    from textworld.envs.pddl.textgen import EvalSymbol

    patch_textworld_eval()
    symbol = EvalSymbol("r.name")
    result = symbol.derive({"variables": {"r": SimpleNamespace(name="kitchen")}})
    assert result[0].symbol == "kitchen"


def test_agentbench_capacity_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr(agentbench_module.requests, "get", lambda *args, **kwargs: Response([0, 1]))
    env = AgentBenchEnvironment("standard", {"protocol": "v02_text"}, 42)
    with pytest.raises(InfrastructureError, match="fewer than requested"):
        env.tasks(300)


def test_agentbench_explicit_task_ids_are_validated(monkeypatch) -> None:
    monkeypatch.setattr(agentbench_module.requests, "get", lambda *args, **kwargs: Response([18, 20, 22]))
    options = {
        "protocol": "v02_text",
        "suite": "dbbench-std",
        "task_ids": ["dbbench-std:22", "dbbench-std:18"],
    }
    env = AgentBenchEnvironment("standard", options, 42)
    assert [task.task_id for task in env.tasks(2)] == ["dbbench-std:22", "dbbench-std:18"]
    env.options["task_ids"] = ["dbbench-std:18"]
    with pytest.raises(InfrastructureError, match="expected tasks=2"):
        env.tasks(2)
    env.options["task_ids"] = ["dbbench-std:18", "dbbench-std:99"]
    with pytest.raises(InfrastructureError, match="Unknown AgentBench task ids"):
        env.tasks(2)


def test_scienceworld_explicit_task_ids_are_validated() -> None:
    env = object.__new__(ScienceWorldEnvironment)
    env.options = {"task_ids": ["boil:21", "find-animal:225"]}
    env.split = "test"

    class Native:
        @staticmethod
        def get_task_names():
            return ["find-animal", "boil"]

        @staticmethod
        def load(*args):
            return None

        @staticmethod
        def get_variations_test():
            return [21, 225]

        get_variations_train = get_variations_test
        get_variations_dev = get_variations_test

    env.native = Native()
    assert [task.task_id for task in env.tasks(2)] == ["boil:21", "find-animal:225"]
    env.options = {"task_ids": ["boil:21"]}
    with pytest.raises(InfrastructureError, match="expected tasks=2"):
        env.tasks(2)
    env.options = {"task_ids": ["boil:21", "missing:1"]}
    with pytest.raises(InfrastructureError, match="Unknown ScienceWorld task ids"):
        env.tasks(2)


def test_task_id_shape() -> None:
    task = TaskSpec("agentbench", "dbbench-std:4", "goal", "standard", 4)
    assert task.native_id == 4


def test_agentbench_v02_preserves_multiline_action_and_uses_official_scorer(monkeypatch) -> None:
    posts = []
    monkeypatch.setattr(agentbench_module.requests, "get", lambda *args, **kwargs: Response([0]))

    def post(url, json, **kwargs):
        posts.append((url, json))
        if url.endswith("start_sample"):
            return Response(
                {
                    "session_id": 8,
                    "output": {
                        "index": 0,
                        "status": "running",
                        "history": [{"role": "user", "content": "Question"}],
                    },
                }
            )
        if url.endswith("interact"):
            return Response(
                {
                    "output": {
                        "index": 0,
                        "status": "completed",
                        "result": {"type": "counting", "answer": "['3']", "error": ""},
                        "history": [],
                    }
                }
            )
        return Response({"custom": {"counting_accuracy": 1.0}})

    monkeypatch.setattr(agentbench_module.requests, "post", post)
    env = AgentBenchEnvironment(
        "standard", {"suite": "dbbench-std", "protocol": "v02_text"}, 42
    )
    task = env.tasks(1)[0]
    env.reset(task)
    action = "Action: Answer\nFinal Answer: [\"3\"]"
    outcome = env.step(action)
    assert posts[1][1]["agent_response"]["content"] == action
    assert posts[2][0].endswith("calculate_overall")
    assert outcome.success and outcome.score == 1.0


def test_agentbench_fc_fails_before_contacting_controller() -> None:
    with pytest.raises(InfrastructureError, match="serialized full OpenAI response"):
        AgentBenchEnvironment("standard", {"protocol": "agentrl_fc"}, 42)


def test_agentbench_local_dbbench_select_and_mutation(tmp_path) -> None:
    records = [
        {
            "description": "Which value belongs to id one?",
            "label": ["alpha"],
            "type": ["other"],
            "add_description": "The table is sample with columns id,value.",
            "table": {
                "table_name": "sample",
                "table_info": {
                    "columns": [{"name": "id"}, {"name": "value"}],
                    "rows": [["1", "alpha"]],
                },
            },
        },
        {
            "description": "Add beta.",
            "label": ["INSERT INTO sample (id, value) VALUES ('2', 'beta')"],
            "type": ["INSERT"],
            "add_description": "The table is sample with columns id,value.",
            "table": {
                "table_name": "sample",
                "table_info": {
                    "columns": [{"name": "id"}, {"name": "value"}],
                    "rows": [["1", "alpha"]],
                },
            },
        },
    ]
    data = tmp_path / "dbbench.jsonl"
    data.write_text("\n".join(__import__("json").dumps(item) for item in records), encoding="utf-8")
    env = AgentBenchEnvironment(
        "standard",
        {"suite": "dbbench-std", "protocol": "local_dbbench", "data_file": str(data)},
        42,
    )
    tasks = env.tasks(2)

    reset = env.reset(tasks[0])
    assert "Which value" in reset.observation
    selected = env.step("Action: Operation\n```sql\nSELECT value FROM sample WHERE id='1';\n```")
    assert not selected.done and "alpha" in selected.observation
    answer = env.step('Action: Answer\nFinal Answer: ["alpha"]')
    assert answer.done and answer.success

    env.reset(tasks[1])
    changed = env.step("Action: Operation\n```sql\nINSERT INTO sample VALUES ('2', 'beta');\n```")
    assert not changed.done
    mutation_answer = env.step('Action: Answer\nFinal Answer: ["done"]')
    assert mutation_answer.done and mutation_answer.success

    env.reset(tasks[1])
    semicolon = env.step(
        "Action: Operation\n```sql\nINSERT INTO sample VALUES ('2', 'beta; still one value');\n```"
    )
    assert not semicolon.done
    env.close()
