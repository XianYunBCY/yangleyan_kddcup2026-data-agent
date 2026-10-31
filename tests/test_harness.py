from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from data_agent_baseline.agents.model import ModelCompletion, ModelMessage, ScriptedModelAdapter
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import (
    ANSWER_VALIDATION_FAILED,
    AnswerTable,
    AnswerValidationError,
    PublicTask,
    TaskAssets,
    TaskRecord,
)
from data_agent_baseline.harness.events import HarnessState
from data_agent_baseline.harness.policy import HarnessPolicy
from data_agent_baseline.harness.replay import replay_trace
from data_agent_baseline.harness.session import HarnessSession
from data_agent_baseline.harness.telemetry import InstrumentedModelAdapter
from data_agent_baseline.harness.tools import HarnessToolRegistry
from data_agent_baseline.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    HarnessConfig,
    RunConfig,
)
from data_agent_baseline.run.runner import run_single_task
from data_agent_baseline.tools.registry import create_default_tool_registry


def _task(root: Path) -> PublicTask:
    context_dir = root / "context"
    context_dir.mkdir(parents=True)
    database_path = context_dir / "numbers.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute("CREATE TABLE numbers (value INTEGER)")
        conn.executemany("INSERT INTO numbers VALUES (?)", [(1,), (2,), (3,)])
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="What is the total?"),
        assets=TaskAssets(task_dir=root, context_dir=context_dir),
    )


def _prepare_input() -> dict[str, object]:
    return {
        "asq": {
            "version": "1",
            "dimensions": [],
            "metrics": ["numbers.total"],
            "filters": [],
            "group_by": [],
            "order_by": [],
            "output": ["SUM(value)"],
        },
        "compiled": {
            "engine": "sqlite",
            "path": "numbers.db",
            "sql": "SELECT SUM(value) FROM numbers",
        },
    }


class MetadataModel:
    def complete_with_metadata(self, messages: list[ModelMessage]) -> ModelCompletion:
        self.messages = messages
        return ModelCompletion(
            content="ok",
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            model="fake-model",
            response_id="response-1",
        )


class HarnessTests(unittest.TestCase):
    def test_answer_table_sanity_validation(self) -> None:
        cases = [
            (AnswerTable(columns=[], rows=[[1]]), "at least one column"),
            (AnswerTable(columns=["value"], rows=[]), "at least one row"),
            (
                AnswerTable(columns=["left", "right"], rows=[[None, None]]),
                "at least one non-NULL value",
            ),
            (
                AnswerTable(columns=["left", "right"], rows=[[1]]),
                "match the number of columns",
            ),
        ]
        for answer, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                self.assertIn(expected_message, answer.validation_error() or "")

        self.assertIsNone(
            AnswerTable(columns=["left", "right"], rows=[[None, 1]]).validation_error()
        )

    def test_raw_terminal_validation_rejects_empty_sql_then_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task = _task(Path(temp_dir))
            session = HarnessSession(task_id=task.task_id, mode="raw", model="fake", max_steps=2)
            tools = HarnessToolRegistry(
                create_default_tool_registry(),
                session,
                HarnessPolicy("raw"),
            )

            with self.assertRaises(AnswerValidationError):
                tools.execute(
                    task,
                    "answer_context_sql",
                    {"path": "numbers.db", "sql": "SELECT value FROM numbers WHERE 0"},
                )

            self.assertNotEqual(session.state, HarnessState.SUBMITTED)
            failed_event = session.events[-1]
            self.assertEqual(failed_event.event_type, "tool.failed")
            self.assertEqual(failed_event.payload["failure_type"], ANSWER_VALIDATION_FAILED)
            self.assertIn("at least one row", failed_event.payload["answer_validation_error"])

            answer = tools.execute(
                task,
                "answer_context_sql",
                {"path": "numbers.db", "sql": "SELECT SUM(value) AS total FROM numbers"},
            )
            self.assertTrue(answer.is_terminal)
            self.assertEqual(answer.answer.rows, [[6]])
            self.assertEqual(session.state, HarnessState.SUBMITTED)

    def test_raw_manual_answer_rejects_all_null_with_failure_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task = _task(Path(temp_dir))
            session = HarnessSession(task_id=task.task_id, mode="raw", model="fake", max_steps=1)
            tools = HarnessToolRegistry(
                create_default_tool_registry(),
                session,
                HarnessPolicy("raw"),
            )

            with self.assertRaises(AnswerValidationError):
                tools.execute(
                    task,
                    "answer",
                    {"columns": ["value"], "rows": [[None]]},
                )

            self.assertEqual(session.events[-1].payload["failure_type"], ANSWER_VALIDATION_FAILED)
            self.assertNotEqual(session.state, HarnessState.SUBMITTED)

    def test_runtime_success_requires_validated_answer(self) -> None:
        result = AgentRunResult(
            task_id="task_1",
            answer=AnswerTable(columns=["value"], rows=[]),
            steps=[],
            failure_reason=None,
            validate_answer_sanity=True,
        )

        self.assertFalse(result.succeeded)
        payload = result.to_dict()
        self.assertEqual(payload["failure_type"], ANSWER_VALIDATION_FAILED)
        self.assertTrue(payload["failure_reason"].startswith(ANSWER_VALIDATION_FAILED))

    def test_raw_runner_recovers_after_rejected_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            task_dir = input_root / "task_1"
            task = _task(task_dir)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "difficulty": task.difficulty,
                        "question": task.question,
                    }
                )
            )
            output_root = root / "runs"
            output_root.mkdir()
            invalid_response = json.dumps(
                {
                    "thought": "No rows yet.",
                    "action": "answer",
                    "action_input": {"columns": ["total"], "rows": []},
                }
            )
            valid_response = json.dumps(
                {
                    "thought": "Submit the corrected result.",
                    "action": "answer",
                    "action_input": {"columns": ["total"], "rows": [[6]]},
                }
            )
            config = AppConfig(
                dataset=DatasetConfig(root_path=input_root),
                agent=AgentConfig(model="scripted", max_steps=2),
                harness=HarnessConfig(mode="raw"),
                run=RunConfig(output_dir=output_root, task_timeout_seconds=0),
            )

            artifact = run_single_task(
                task_id="task_1",
                config=config,
                run_output_dir=output_root,
                model=ScriptedModelAdapter([invalid_response, valid_response]),
            )

            self.assertTrue(artifact.succeeded)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["answer"]["rows"], [[6]])
            self.assertEqual(
                trace["steps"][0]["observation"]["failure_type"],
                ANSWER_VALIDATION_FAILED,
            )
            self.assertIsNone(trace["answer_validation_error"])

    def test_raw_runner_labels_terminal_answer_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            task_dir = input_root / "task_1"
            task = _task(task_dir)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "difficulty": task.difficulty,
                        "question": task.question,
                    }
                )
            )
            output_root = root / "runs"
            output_root.mkdir()
            invalid_response = json.dumps(
                {
                    "thought": "Submit an empty result.",
                    "action": "answer",
                    "action_input": {"columns": ["total"], "rows": []},
                }
            )
            config = AppConfig(
                dataset=DatasetConfig(root_path=input_root),
                agent=AgentConfig(model="scripted", max_steps=1),
                harness=HarnessConfig(mode="raw"),
                run=RunConfig(output_dir=output_root, task_timeout_seconds=0),
            )

            artifact = run_single_task(
                task_id="task_1",
                config=config,
                run_output_dir=output_root,
                model=ScriptedModelAdapter([invalid_response] * 4),
            )

            self.assertFalse(artifact.succeeded)
            self.assertIsNone(artifact.prediction_csv_path)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["failure_type"], ANSWER_VALIDATION_FAILED)
            self.assertTrue(trace["failure_reason"].startswith(ANSWER_VALIDATION_FAILED))
            self.assertIn("at least one row", trace["answer_validation_error"])

    def test_model_metadata_is_recorded(self) -> None:
        session = HarnessSession(task_id="task_1", mode="raw", model="fake", max_steps=1)
        adapter = InstrumentedModelAdapter(MetadataModel(), session)
        self.assertEqual(adapter.complete([ModelMessage(role="user", content="hello")]), "ok")
        totals = session.to_dict()["totals"]
        self.assertEqual(totals["model_calls"], 1)
        self.assertEqual(totals["model_calls_with_token_usage"], 1)
        self.assertEqual(totals["prompt_tokens"], 10)
        self.assertEqual(totals["completion_tokens"], 2)
        self.assertEqual(totals["total_tokens"], 12)

    def test_semantic_mode_requires_prepared_exact_sql(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task = _task(Path(temp_dir))
            session = HarnessSession(
                task_id=task.task_id, mode="semantic", model="fake", max_steps=4
            )
            tools = HarnessToolRegistry(
                create_default_tool_registry(),
                session,
                HarnessPolicy("semantic"),
            )
            query_input = {"path": "numbers.db", "sql": "SELECT SUM(value) FROM numbers"}
            with self.assertRaises(PermissionError):
                tools.execute(task, "answer_context_sql", query_input)

            prepared = tools.execute(task, "prepare_asq", _prepare_input())
            self.assertTrue(prepared.ok)
            with self.assertRaises(PermissionError):
                tools.execute(
                    task,
                    "answer_context_sql",
                    {"path": "numbers.db", "sql": "SELECT COUNT(*) FROM numbers"},
                )
            answer = tools.execute(task, "answer_context_sql", query_input)
            self.assertTrue(answer.is_terminal)
            self.assertEqual(answer.answer.rows, [[6]])

    def test_asq_and_sql_trace_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task = _task(root)
            semantic_session = HarnessSession(
                task_id=task.task_id,
                mode="semantic",
                model="fake",
                max_steps=4,
            )
            semantic_tools = HarnessToolRegistry(
                create_default_tool_registry(),
                semantic_session,
                HarnessPolicy("semantic"),
            )
            semantic_tools.execute(task, "prepare_asq", _prepare_input())
            semantic_session.finalize(succeeded=True)
            trace_path = root / "trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "steps": [
                            {
                                "step_index": 2,
                                "action": "answer_context_sql",
                                "action_input": {
                                    "path": "numbers.db",
                                    "sql": "SELECT SUM(value) FROM numbers",
                                },
                            }
                        ],
                        "harness": semantic_session.to_dict(),
                    }
                )
            )

            asq_result = replay_trace(task=task, trace_path=trace_path, kind="asq")
            self.assertTrue(asq_result["succeeded"])
            self.assertEqual(asq_result["answer"]["rows"], [[6]])
            self.assertEqual(asq_result["harness"]["totals"]["model_calls"], 0)

            sql_result = replay_trace(task=task, trace_path=trace_path, kind="sql")
            self.assertTrue(sql_result["succeeded"])
            self.assertEqual(sql_result["answer"]["rows"], [[6]])

    def test_semantic_runner_writes_harness_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            task_dir = input_root / "task_1"
            task = _task(task_dir)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "difficulty": task.difficulty,
                        "question": task.question,
                    }
                )
            )
            output_root = root / "runs"
            output_root.mkdir()
            prepare_response = json.dumps(
                {
                    "thought": "Prepare the semantic query.",
                    "action": "prepare_asq",
                    "action_input": _prepare_input(),
                }
            )
            answer_response = json.dumps(
                {
                    "thought": "Execute the approved query.",
                    "action": "answer_context_sql",
                    "action_input": {
                        "path": "numbers.db",
                        "sql": "SELECT SUM(value) FROM numbers",
                    },
                }
            )
            config = AppConfig(
                dataset=DatasetConfig(root_path=input_root),
                agent=AgentConfig(model="scripted", max_steps=2),
                harness=HarnessConfig(
                    mode="semantic",
                    prompt_version="test-prompt",
                    semantic_model_version="test-catalog",
                ),
                run=RunConfig(output_dir=output_root, task_timeout_seconds=0),
            )
            artifact = run_single_task(
                task_id="task_1",
                config=config,
                run_output_dir=output_root,
                model=ScriptedModelAdapter([prepare_response, answer_response]),
            )
            self.assertTrue(artifact.succeeded)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["answer"]["rows"], [[6]])
            self.assertEqual(trace["harness"]["state"], "submitted")
            self.assertEqual(trace["harness"]["prepared_plan_count"], 1)
            self.assertEqual(trace["harness"]["totals"]["model_calls"], 2)
            self.assertEqual(trace["harness"]["versions"]["semantic_model"], "test-catalog")


if __name__ == "__main__":
    unittest.main()
