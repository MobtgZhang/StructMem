from __future__ import annotations

import ast
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import requests

from structmem.environments.base import BenchmarkEnvironment, InfrastructureError, ResetResult, StepResult
from structmem.types import TaskSpec


DBBENCH_PROMPT = """I will ask you a question, then you should help me operate a SQL database to answer the question.
Every round, choose exactly one action in the following format.

To execute one SQL statement:
Action: Operation
```sql
SELECT * FROM `table`;
```

To submit the final answer:
Action: Answer
Final Answer: ["ANSWER1", "ANSWER2", ...]

Execute at most one statement per operation. Return answers exactly as they appear in the database.
For INSERT, DELETE, or UPDATE tasks, submit any final answer only after the database modification is complete.
"""


class AgentBenchEnvironment(BenchmarkEnvironment):
    """AgentBench DBBench using either its v0.2 controller or a local SQL backend."""

    def __init__(self, split: str, options: dict[str, Any], seed: int):
        self.split = split
        self.options = options
        self.suite = str(options.get("suite", "dbbench-std"))
        self.timeout = float(options.get("timeout_seconds", 30))
        self.protocol = str(options.get("protocol", ""))
        self.session_id: str | None = None
        self.latest: dict[str, Any] = {}
        self._connection: sqlite3.Connection | None = None
        self._entry: dict[str, Any] | None = None
        self._round = 0
        self._max_round = int(options.get("max_round", 15))

        if self.protocol == "local_dbbench":
            data_file = options.get("data_file") or _default_data_file(self.suite)
            self.data_file = Path(str(data_file)).expanduser().resolve()
            if not self.data_file.exists():
                raise InfrastructureError(f"AgentBench DBBench data file not found: {self.data_file}")
            try:
                self.dataset = [json.loads(line) for line in self.data_file.read_text(encoding="utf-8").splitlines() if line]
            except Exception as exc:
                raise InfrastructureError(f"Cannot load AgentBench DBBench data {self.data_file}: {exc}") from exc
            self.indices = list(range(len(self.dataset)))
            return

        if self.protocol == "agentrl_fc":
            raise InfrastructureError(
                "AgentBench agentrl_fc is unsupported because its serialized full OpenAI response transport is not published"
            )
        if self.protocol != "v02_text":
            raise InfrastructureError(
                "AgentBench options.protocol must be local_dbbench or v02_text"
            )
        self.controller = str(options.get("controller", "http://localhost:5000/api")).rstrip("/")
        try:
            response = requests.get(f"{self.controller}/get_indices", params={"name": self.suite}, timeout=self.timeout)
            response.raise_for_status()
            self.indices = list(response.json())
        except Exception as exc:
            raise InfrastructureError(
                f"AgentBench controller/suite unavailable at {self.controller} ({self.suite}): {exc}"
            ) from exc

    def tasks(self, limit: int) -> list[TaskSpec]:
        tasks = [
            TaskSpec(
                dataset="agentbench",
                task_id=f"{self.suite}:{index}",
                goal=(self.dataset[index]["description"] if self.protocol == "local_dbbench" else "provided by AgentBench session"),
                split=self.split,
                native_id=index,
                metadata={"suite": self.suite, "protocol": self.protocol},
            )
            for index in self.indices
        ]
        selected_ids = self.options.get("task_ids")
        if selected_ids is not None:
            if not isinstance(selected_ids, list) or any(not isinstance(item, str) for item in selected_ids):
                raise InfrastructureError("AgentBench options.task_ids must be a list of task-id strings")
            if len(selected_ids) != limit:
                raise InfrastructureError(
                    f"AgentBench options.task_ids has {len(selected_ids)} entries, expected tasks={limit}"
                )
            by_id = {task.task_id: task for task in tasks}
            missing = [task_id for task_id in selected_ids if task_id not in by_id]
            if missing:
                raise InfrastructureError(f"Unknown AgentBench task ids: {missing}")
            return [by_id[task_id] for task_id in selected_ids]
        if len(tasks) < limit:
            raise InfrastructureError(
                f"AgentBench suite {self.suite} has {len(tasks)} tasks, fewer than requested {limit}"
            )
        return tasks[:limit]

    def reset(self, task: TaskSpec) -> ResetResult:
        if self.protocol == "local_dbbench":
            self._close_connection()
            self._entry = self.dataset[int(task.native_id)]
            self._connection = _create_database(self._entry)
            self._round = 0
            prompt = _task_prompt(self._entry)
            return ResetResult(observation=prompt, info={"goal": prompt, "status": "running"})
        try:
            response = requests.post(
                f"{self.controller}/start_sample",
                json={"name": self.suite, "index": task.native_id},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise InfrastructureError(f"AgentBench start_sample request failed: {exc}") from exc
        if response.status_code != 200:
            raise InfrastructureError(f"AgentBench start_sample failed ({response.status_code}): {response.text[:500]}")
        payload = response.json()
        self.session_id = payload["session_id"]
        self.latest = payload["output"]
        observation = _new_environment_messages(self.latest.get("history", []))
        return ResetResult(observation=observation, info={"goal": observation, "status": self.latest.get("status")})

    def step(self, action: str) -> StepResult:
        if self.protocol == "local_dbbench":
            return self._local_step(action)
        if self.session_id is None:
            raise InfrastructureError("AgentBench step called before reset")
        try:
            response = requests.post(
                f"{self.controller}/interact",
                json={"session_id": self.session_id, "agent_response": {"status": "normal", "content": action}},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise InfrastructureError(f"AgentBench interact request failed: {exc}") from exc
        if response.status_code != 200:
            raise InfrastructureError(f"AgentBench interact failed ({response.status_code}): {response.text[:500]}")
        payload = response.json()
        self.latest = payload["output"]
        status = str(self.latest.get("status", "")).lower()
        done = status != "running"
        reward, score = self._score_output() if done else (0.0, 0.0)
        return StepResult(
            observation=_new_environment_messages(self.latest.get("history", [])),
            reward=reward,
            done=done,
            success=done and reward > 0,
            score=score,
            info={"status": status, "raw_output": self.latest},
        )

    def _local_step(self, action: str) -> StepResult:
        if self._connection is None or self._entry is None:
            raise InfrastructureError("AgentBench local step called before reset")
        self._round += 1
        operation = re.search(r"Action:\s*Operation", action, re.IGNORECASE)
        answer_action = re.search(r"Action:\s*Answer", action, re.IGNORECASE)
        if operation:
            block = re.search(r"```sql\s*\n?([\s\S]*?)```", action, re.IGNORECASE)
            if not block:
                return self._local_finish(False, "invalid_action", "Missing SQL code block")
            sql = block.group(1).strip()
            try:
                cursor = self._connection.execute(_sqlite_compatible_sql(sql, self._entry))
                rows = cursor.fetchall() if cursor.description else []
                self._connection.commit()
                observation = str([tuple(row) for row in rows])
            except sqlite3.Error as exc:
                observation = f"SQL error: {exc}"
            if self._round >= self._max_round:
                return self._local_finish(False, "task_limit_reached", observation)
            return StepResult(observation, 0.0, False, False, 0.0, {"status": "running"})
        if answer_action:
            match = re.search(r"Final Answer:\s*([\s\S]+)$", action, re.IGNORECASE)
            if not match:
                return self._local_finish(False, "invalid_action", "Missing Final Answer")
            submitted = _parse_answer(match.group(1).strip())
            correct = _score_entry(self._connection, self._entry, submitted)
            return self._local_finish(correct, "completed", "Correct" if correct else "Incorrect")
        return self._local_finish(False, "invalid_action", "Expected Action: Operation or Action: Answer")

    def _local_finish(self, correct: bool, status: str, observation: str) -> StepResult:
        score = float(correct)
        return StepResult(
            observation=observation,
            reward=score,
            done=True,
            success=correct,
            score=score,
            info={"status": status, "is_correct": correct, "type": (self._entry or {}).get("type", [""])[0]},
        )

    def _score_output(self) -> tuple[float, float]:
        reward, score = _reward_score(self.latest)
        if isinstance(self.latest.get("result"), dict) and "is_correct" in self.latest["result"]:
            return reward, score
        try:
            response = requests.post(
                f"{self.controller}/calculate_overall",
                json={"name": self.suite, "results": [self.latest]},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise InfrastructureError(f"AgentBench calculate_overall request failed: {exc}") from exc
        if response.status_code != 200:
            raise InfrastructureError(f"AgentBench calculate_overall failed ({response.status_code}): {response.text[:500]}")
        custom = response.json().get("custom", response.json())
        task_type = str((self.latest.get("result") or {}).get("type", ""))
        metric = f"{task_type}_accuracy"
        if metric not in custom:
            raise InfrastructureError(f"AgentBench v0.2 scorer did not return {metric!r}; available={sorted(custom)}")
        value = float(custom[metric])
        return value, value

    def close(self) -> None:
        self._close_connection()
        if self.protocol == "local_dbbench" or self.session_id is None or str(self.latest.get("status", "")).lower() != "running":
            return
        try:
            requests.post(f"{self.controller}/cancel", json={"session_id": self.session_id}, timeout=self.timeout)
        except requests.RequestException:
            pass
        finally:
            self.session_id = None

    def _close_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def _default_data_file(suite: str) -> Path:
    filename = "train.jsonl" if suite in {"dbbench-env_train", "dbbench-train"} else "standard.jsonl"
    return Path(__file__).resolve().parents[1] / "benchmark_data" / filename


def _task_prompt(entry: dict[str, Any]) -> str:
    extra = entry.get("add_description", "")
    return f"{DBBENCH_PROMPT}\nQuestion: {entry['description']}\n{extra}".strip()


def _tables(entry: dict[str, Any]) -> list[dict[str, Any]]:
    value = entry["table"]
    return value if isinstance(value, list) else [value]


def _create_database(entry: dict[str, Any]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    for table in _tables(entry):
        name = table["table_name"]
        columns = table["table_info"]["columns"]
        definitions = ",".join(f'`{column["name"].replace("`", "``")}` TEXT' for column in columns)
        escaped_name = name.replace('"', '""')
        connection.execute(f'CREATE TABLE "{escaped_name}" ({definitions})')
        rows = table["table_info"]["rows"]
        if rows:
            placeholders = ",".join("?" for _ in columns)
            connection.executemany(f'INSERT INTO "{escaped_name}" VALUES ({placeholders})', rows)
    connection.commit()
    return connection


def _parse_answer(raw: str) -> list[Any]:
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw.strip().strip('"\'')]
    return list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]


def _normalize(value: Any) -> str:
    if value is None:
        return "0"
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if "," in text and not text.startswith("["):
        text = text.replace(",", "")
    return "0" if text.lower() in {"none", "null", "undefined", "nan", "inf", "infinity", "-inf", "-infinity", ""} else text


def _answers_equal(answer: list[Any], ground_truth: list[Any]) -> bool:
    left = [_normalize(value) for value in answer]
    right = [_normalize(value) for value in ground_truth]
    if len(left) != len(right):
        return False
    if len(left) == 1:
        try:
            return abs(float(left[0]) - float(right[0])) <= 1e-2
        except ValueError:
            return left[0] == right[0]
    return sorted(left) == sorted(right)


def _database_state(connection: sqlite3.Connection, entry: dict[str, Any]) -> list[tuple[str, tuple[tuple[str, ...], ...]]]:
    state = []
    for table in _tables(entry):
        name = table["table_name"]
        escaped_name = name.replace('"', '""')
        rows = connection.execute(f'SELECT * FROM "{escaped_name}"').fetchall()
        state.append((name, tuple(sorted(tuple(_normalize(value) for value in row) for row in rows))))
    return sorted(state)


def _score_entry(connection: sqlite3.Connection, entry: dict[str, Any], submitted: list[Any]) -> bool:
    task_type = entry.get("type", ["other"])[0]
    if task_type not in {"INSERT", "DELETE", "UPDATE"}:
        return _answers_equal(submitted, entry.get("label", []))
    expected = _create_database(entry)
    try:
        expected_sql = (entry.get("sql") or {}).get("query")
        if not expected_sql:
            labels = entry.get("label") or []
            expected_sql = labels[0] if labels else None
        if not expected_sql:
            raise InfrastructureError("DBBench mutation task has neither official SQL nor a labeled SQL statement")
        expected.execute(_sqlite_compatible_sql(expected_sql, entry))
        expected.commit()
        return _database_state(connection, entry) == _database_state(expected, entry)
    except sqlite3.Error as exc:
        raise InfrastructureError(f"Official DBBench SQL is not SQLite-compatible: {exc}") from exc
    finally:
        expected.close()


def _sqlite_compatible_sql(sql: str, entry: dict[str, Any]) -> str:
    """Quote dataset table identifiers that SQLite cannot parse unquoted."""
    converted = sql
    for table in _tables(entry):
        name = str(table["table_name"])
        if re.match(r"^[A-Za-z_]\w*$", name):
            continue
        quoted = f'"{name.replace(chr(34), chr(34) * 2)}"'
        converted = re.sub(
            rf"(?i)(?<![`\"\w]){re.escape(name)}(?![`\"\w])",
            quoted,
            converted,
        )
    return converted


def _new_environment_messages(history: list[dict[str, Any]]) -> str:
    messages = []
    for item in history:
        if item.get("role") in {"system", "user", "tool"}:
            messages.append(f"[{item.get('role')}]: {item.get('content', '')}")
    return "\n".join(messages[-8:])


def _reward_score(output: dict[str, Any]) -> tuple[float, float]:
    result = output.get("result")
    if isinstance(result, dict) and "is_correct" in result:
        value = float(bool(result["is_correct"]))
        return value, value
    for item in reversed(output.get("history", [])):
        if "reward" in item or "score" in item:
            return float(item.get("reward", 0.0)), float(item.get("score", item.get("reward", 0.0)))
    return 0.0, 0.0
