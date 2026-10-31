from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.harness.policy import HarnessPolicy
from data_agent_baseline.harness.replay import replay_trace
from data_agent_baseline.harness.session import HarnessSession
from data_agent_baseline.harness.tools import HarnessToolRegistry
from data_agent_baseline.semantic.catalog import load_semantic_catalog
from data_agent_baseline.semantic.compiler import compile_asq, parse_asq
from data_agent_baseline.semantic.executor import execute_semantic_plan
from data_agent_baseline.tools.registry import create_default_tool_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _task_169():
    return DABenchPublicDataset(PROJECT_ROOT / "public" / "input").get_task("task_169")


def _task_169_asq() -> dict[str, object]:
    return {
        "version": "1",
        "dimensions": [],
        "metrics": ["debit_card.average_monthly_consumption"],
        "filters": [
            {"field": "customer.segment", "op": "=", "value": "SME"},
            {"field": "consumption.year", "op": "=", "value": 2013},
        ],
        "group_by": [],
        "order_by": [],
        "output": ["average_monthly_consumption"],
    }


def _task_269():
    return DABenchPublicDataset(PROJECT_ROOT / "public" / "input").get_task("task_269")


def _task_269_asq() -> dict[str, object]:
    return {
        "version": "1",
        "dimensions": ["hero.name"],
        "metrics": [],
        "filters": [
            {"field": "power.name", "op": "=", "value": "Death Touch"},
        ],
        "group_by": [],
        "order_by": [],
        "output": ["superhero_name"],
    }


def _task_287():
    return DABenchPublicDataset(PROJECT_ROOT / "public" / "input").get_task("task_287")


def _task_287_asq() -> dict[str, object]:
    return {
        "version": "1",
        "dimensions": ["hero.gender"],
        "metrics": [],
        "filters": [
            {"field": "power.name", "op": "=", "value": "Phoenix Force"},
        ],
        "group_by": [],
        "order_by": [],
        "output": ["gender"],
    }


def _match_model(task, asq_payload: dict[str, object]):
    asq = parse_asq(asq_payload)
    return load_semantic_catalog().match_task(
        task,
        required_dimensions=set(asq.dimensions) | set(asq.group_by),
        required_metrics=asq.metrics,
        required_filter_fields={str(item.get("field", "")) for item in asq.filters},
        required_order_fields={str(item.get("field", "")) for item in asq.order_by},
    )


class SemanticCompilerTests(unittest.TestCase):
    def test_task_169_compiles_and_matches_gold_exactly(self) -> None:
        task = _task_169()
        model = _match_model(task, _task_169_asq())
        self.assertIsNotNone(model)
        plan = compile_asq(model, _task_169_asq())
        result = execute_semantic_plan(task, plan, limit=10)
        self.assertEqual(plan.semantic_model_id, "debit_card")
        self.assertEqual(plan.semantic_model_version, "debit-card-v1")
        self.assertEqual(
            result["columns"],
            ["AVG(T2.Consumption) / 12"],
        )
        self.assertEqual(result["rows"], [[459.9562642871061]])
        self.assertEqual(len(plan.join_path), 1)
        self.assertEqual(plan.join_path[0].relationship, "one_to_many")

    def test_task_269_compiles_three_sources_and_matches_gold_exactly(self) -> None:
        task = _task_269()
        model = _match_model(task, _task_269_asq())
        self.assertIsNotNone(model)
        plan = compile_asq(model, _task_269_asq())
        result = execute_semantic_plan(task, plan, limit=20)
        self.assertEqual(plan.semantic_model_id, "superhero")
        self.assertEqual(plan.semantic_model_version, "superhero-v1")
        self.assertEqual([source.kind for source in plan.sources], ["sqlite", "csv", "json"])
        self.assertEqual(result["columns"], ["superhero_name"])
        self.assertEqual(
            result["rows"],
            [
                ["Black Flash"],
                ["Blackwulf"],
                ["Hela"],
                ["Living Tribunal"],
                ["One-Above-All"],
                ["Poison Ivy"],
                ["Spectre"],
            ],
        )
        self.assertEqual(
            [join.relationship for join in plan.join_path],
            ["one_to_many", "many_to_one"],
        )

    def test_task_287_rejects_source_overlap_without_gender_capability(self) -> None:
        task = _task_287()
        catalog = load_semantic_catalog()
        candidates = catalog.candidate_models(task)
        self.assertEqual([model.model_id for model in candidates], ["superhero"])

        asq = parse_asq(_task_287_asq())
        requirements = {
            "required_dimensions": set(asq.dimensions) | set(asq.group_by),
            "required_metrics": asq.metrics,
            "required_filter_fields": {str(item.get("field", "")) for item in asq.filters},
            "required_order_fields": {str(item.get("field", "")) for item in asq.order_by},
        }
        self.assertIsNone(catalog.match_task(task, **requirements))
        decision = next(
            item
            for item in catalog.evaluate_task(task, **requirements)
            if item.model.model_id == "superhero"
        )
        self.assertTrue(decision.source_compatible)
        self.assertFalse(decision.matched)
        self.assertEqual(decision.missing_dimensions, ("hero.gender",))
        self.assertIn("missing dimensions: hero.gender", decision.reasons)

        session = HarnessSession(
            task_id=task.task_id,
            mode="semantic",
            model="fake",
            max_steps=4,
        )
        tools = HarnessToolRegistry(
            create_default_tool_registry(),
            session,
            HarnessPolicy("semantic"),
        )
        source_catalog = tools.execute(task, "get_semantic_catalog", {})
        self.assertEqual(source_catalog.content["match_status"], "source_candidate")
        with self.assertRaisesRegex(
            ValueError,
            "missing dimensions: hero.gender",
        ):
            tools.execute(task, "prepare_asq", {"asq": _task_287_asq()})

        rejection = next(
            event for event in session.events if event.event_type == "semantic.match.rejected"
        )
        self.assertEqual(rejection.payload["requirements"]["dimensions"], ["hero.gender"])
        self.assertEqual(
            rejection.payload["candidates"][0]["missing_dimensions"],
            ["hero.gender"],
        )

    def test_task_269_compiler_gate_and_replay(self) -> None:
        task = _task_269()
        session = HarnessSession(task_id=task.task_id, mode="semantic", model="fake", max_steps=4)
        tools = HarnessToolRegistry(
            create_default_tool_registry(),
            session,
            HarnessPolicy("semantic"),
        )
        catalog = tools.execute(task, "get_semantic_catalog", {})
        self.assertEqual(catalog.content["model_id"], "superhero")
        with self.assertRaises(PermissionError):
            tools.execute(task, "answer_duckdb", {"sql": "SELECT 1"})

        prepared = tools.execute(task, "prepare_asq", {"asq": _task_269_asq()})
        answer = tools.execute(
            task,
            "answer_prepared_asq",
            {"fingerprint": prepared.content["fingerprint"]},
        )
        self.assertTrue(answer.is_terminal)
        self.assertEqual(answer.answer.columns, ["superhero_name"])
        self.assertEqual(len(answer.answer.rows), 7)

        trace = {
            "task_id": task.task_id,
            "steps": [
                {
                    "step_index": 1,
                    "action": "prepare_asq",
                    "action_input": {"asq": _task_269_asq()},
                },
                {
                    "step_index": 2,
                    "action": "answer_prepared_asq",
                    "action_input": {"fingerprint": prepared.content["fingerprint"]},
                },
            ],
            "harness": session.to_dict(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.json"
            trace_path.write_text(json.dumps(trace))
            asq_result = replay_trace(task=task, trace_path=trace_path, kind="asq")
            sql_result = replay_trace(task=task, trace_path=trace_path, kind="sql")

        for result in (asq_result, sql_result):
            self.assertTrue(result["succeeded"])
            self.assertEqual(result["answer"]["columns"], ["superhero_name"])
            self.assertEqual(len(result["answer"]["rows"]), 7)
            self.assertEqual(result["harness"]["totals"]["model_calls"], 0)

    def test_compiler_backed_task_rejects_model_sql_and_manual_answer(self) -> None:
        task = _task_169()
        session = HarnessSession(task_id=task.task_id, mode="semantic", model="fake", max_steps=4)
        tools = HarnessToolRegistry(
            create_default_tool_registry(),
            session,
            HarnessPolicy("semantic"),
        )
        catalog = tools.execute(task, "get_semantic_catalog", {})
        self.assertEqual(catalog.content["model_id"], "debit_card")
        with self.assertRaises(PermissionError):
            tools.execute(
                task,
                "answer_duckdb",
                {"sql": "SELECT 1"},
            )
        with self.assertRaises(PermissionError):
            tools.execute(task, "answer", {"columns": ["x"], "rows": [[1]]})
        with self.assertRaises(ValueError):
            tools.execute(
                task,
                "prepare_asq",
                {
                    "asq": _task_169_asq(),
                    "compiled": {"engine": "duckdb", "sql": "SELECT 1"},
                },
            )

        prepared = tools.execute(task, "prepare_asq", {"asq": _task_169_asq()})
        self.assertEqual(prepared.content["status"], "compiled")
        self.assertNotIn("sql", prepared.content)
        answer = tools.execute(
            task,
            "answer_prepared_asq",
            {"fingerprint": prepared.content["fingerprint"]},
        )
        self.assertTrue(answer.is_terminal)
        self.assertEqual(answer.answer.columns, ["AVG(T2.Consumption) / 12"])
        self.assertEqual(answer.answer.rows, [[459.9562642871061]])

    def test_asq_replay_recompiles_and_sql_replay_uses_recorded_plan(self) -> None:
        task = _task_169()
        session = HarnessSession(task_id=task.task_id, mode="semantic", model="fake", max_steps=4)
        tools = HarnessToolRegistry(
            create_default_tool_registry(),
            session,
            HarnessPolicy("semantic"),
        )
        prepared = tools.execute(task, "prepare_asq", {"asq": _task_169_asq()})
        trace = {
            "task_id": task.task_id,
            "steps": [
                {
                    "step_index": 1,
                    "action": "prepare_asq",
                    "action_input": {"asq": _task_169_asq()},
                },
                {
                    "step_index": 2,
                    "action": "answer_prepared_asq",
                    "action_input": {"fingerprint": prepared.content["fingerprint"]},
                },
            ],
            "harness": session.to_dict(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.json"
            trace_path.write_text(json.dumps(trace))
            asq_result = replay_trace(task=task, trace_path=trace_path, kind="asq")
            sql_result = replay_trace(task=task, trace_path=trace_path, kind="sql")

        for result in (asq_result, sql_result):
            self.assertTrue(result["succeeded"])
            self.assertEqual(result["answer"]["rows"], [[459.9562642871061]])
            self.assertEqual(result["harness"]["totals"]["model_calls"], 0)
        asq_events = [event["event_type"] for event in asq_result["harness"]["events"]]
        self.assertIn("asq.plan.compiled", asq_events)
        sql_events = [event["event_type"] for event in sql_result["harness"]["events"]]
        self.assertNotIn("asq.plan.compiled", sql_events)


if __name__ == "__main__":
    unittest.main()
