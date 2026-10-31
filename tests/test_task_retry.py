from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI as SDKOpenAI

from data_agent_baseline.agents import model as model_module
from data_agent_baseline.agents.model import OpenAIModelAdapter, ScriptedModelAdapter
from data_agent_baseline.benchmark.schema import ANSWER_VALIDATION_FAILED
from data_agent_baseline.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    HarnessConfig,
    RunConfig,
)
from data_agent_baseline.run import runner


TASK_ID = "task_1"


def _write_task(input_root: Path) -> None:
    task_dir = input_root / TASK_ID
    (task_dir / "context").mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "difficulty": "easy",
                "question": "Return one value.",
            }
        )
    )


def _config(root: Path, *, timeout_seconds: int = 10, max_steps: int = 1) -> AppConfig:
    input_root = root / "input"
    _write_task(input_root)
    return AppConfig(
        dataset=DatasetConfig(root_path=input_root),
        agent=AgentConfig(model="scripted", max_steps=max_steps),
        harness=HarnessConfig(mode="raw"),
        run=RunConfig(
            output_dir=root / "runs",
            task_timeout_seconds=timeout_seconds,
        ),
    )


def _result_payload(
    config: AppConfig,
    *,
    attempt_number: int,
    prior_failure_types: tuple[str, ...],
    succeeded: bool,
    failure_type: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    session = runner._build_harness_session(
        task_id=TASK_ID,
        config=config,
        tool_descriptions="",
        attempt_number=attempt_number,
        prior_failure_types=prior_failure_types,
    )
    session.finalize(succeeded=succeeded, failure_reason=failure_reason)
    return {
        "task_id": TASK_ID,
        "answer": {"columns": ["value"], "rows": [[1]]} if succeeded else None,
        "steps": [],
        "failure_reason": failure_reason,
        "failure_type": failure_type,
        "succeeded": succeeded,
        "harness": session.to_dict(),
    }


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value


class AlwaysFailingModel:
    def __init__(self, message: str) -> None:
        self.message = message
        self.call_count = 0

    def complete(self, messages) -> str:
        del messages
        self.call_count += 1
        raise RuntimeError(self.message)


class TaskRetryTests(unittest.TestCase):
    def test_connection_error_retries_after_delay_and_records_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=0)

            def run_attempt(**kwargs):
                if kwargs["attempt_number"] == 1:
                    return _result_payload(
                        config,
                        attempt_number=1,
                        prior_failure_types=kwargs["prior_failure_types"],
                        succeeded=False,
                        failure_reason=(
                            "__model_error__: Model request failed: Connection error."
                        ),
                    )
                return _result_payload(
                    config,
                    attempt_number=2,
                    prior_failure_types=kwargs["prior_failure_types"],
                    succeeded=True,
                )

            with (
                patch.object(runner, "_run_single_task_core", side_effect=run_attempt) as mocked,
                patch.object(runner, "sleep") as mocked_sleep,
            ):
                artifact = runner.run_single_task(
                    task_id=TASK_ID,
                    config=config,
                    run_output_dir=config.run.output_dir,
                    model=object(),
                )

            self.assertTrue(artifact.succeeded)
            self.assertEqual(mocked.call_count, 2)
            mocked_sleep.assert_called_once_with(
                runner.MODEL_CONNECTION_RETRY_DELAY_SECONDS
            )
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["attempt_count"], 2)
            self.assertEqual(trace["attempts"][0]["failure_type"], runner.TASK_CRASH_FAILURE)
            self.assertEqual(
                trace["attempts"][1]["prior_failure_types"],
                [runner.TASK_CRASH_FAILURE],
            )

    def test_request_timeout_model_error_is_classified_as_task_crash(self) -> None:
        run_result = {
            "succeeded": False,
            "failure_reason": "__model_error__: Model request failed: Request timed out.",
            "failure_type": None,
        }

        self.assertEqual(
            runner._attempt_failure_type(run_result), runner.TASK_CRASH_FAILURE
        )

    def test_non_connection_model_errors_keep_two_session_attempts(self) -> None:
        messages = (
            "Model request failed: Error code: 401 - Unauthorized.",
            "Model response missing choices.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, message in enumerate(messages):
                with self.subTest(message=message):
                    case_root = root / str(index)
                    config = _config(case_root, timeout_seconds=0, max_steps=4)
                    model = AlwaysFailingModel(message)

                    run_result = runner._run_single_task_core(
                        task_id=TASK_ID,
                        config=config,
                        model=model,
                    )

                    self.assertEqual(model.call_count, 2)
                    self.assertTrue(str(run_result["failure_reason"]).startswith("__model_error__:"))
                    self.assertIsNone(runner._attempt_failure_type(run_result))

    def test_non_connection_model_error_does_not_sleep_or_task_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=0)
            failure = _result_payload(
                config,
                attempt_number=1,
                prior_failure_types=(),
                succeeded=False,
                failure_reason="__model_error__: Model request failed: Error code: 429.",
            )

            with (
                patch.object(runner, "_run_single_task_core", return_value=failure) as mocked,
                patch.object(runner, "sleep") as mocked_sleep,
            ):
                artifact = runner.run_single_task(
                    task_id=TASK_ID,
                    config=config,
                    run_output_dir=config.run.output_dir,
                    model=object(),
                )

            self.assertFalse(artifact.succeeded)
            self.assertEqual(mocked.call_count, 1)
            mocked_sleep.assert_not_called()

    def test_connection_failure_uses_four_mock_http_requests_per_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=0, max_steps=4)
            requests: list[httpx.Request] = []

            def fail_connection(request: httpx.Request) -> httpx.Response:
                requests.append(request)
                raise httpx.ConnectError("temporary connection failure", request=request)

            http_client = httpx.Client(transport=httpx.MockTransport(fail_connection))
            sdk_client = SDKOpenAI(
                api_key="test-key",
                base_url="https://model.invalid/v1",
                max_retries=model_module.MODEL_MAX_RETRIES,
                http_client=http_client,
            )
            adapter = OpenAIModelAdapter(
                model="test-model",
                api_base="https://model.invalid/v1",
                api_key="test-key",
                temperature=0.0,
            )

            def openai_factory(**kwargs):
                self.assertEqual(kwargs["max_retries"], 1)
                return sdk_client

            try:
                with (
                    patch.object(model_module, "OpenAI", side_effect=openai_factory),
                    patch.object(model_module.time, "sleep"),
                ):
                    run_result = runner._run_single_task_core(
                        task_id=TASK_ID,
                        config=config,
                        model=adapter,
                    )
            finally:
                sdk_client.close()

            self.assertEqual(len(requests), 4)
            self.assertEqual(len(run_result["steps"]), 1)
            self.assertEqual(run_result["steps"][0]["action"], "__model_error__")
            self.assertEqual(
                run_result["failure_reason"],
                "__model_error__: Model request failed: Connection error.",
            )

    def test_timeout_retries_and_second_attempt_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root)

            def run_attempt(**kwargs):
                attempt_number = kwargs["attempt_number"]
                prior = kwargs["prior_failure_types"]
                if attempt_number == 1:
                    failed = _result_payload(
                        config,
                        attempt_number=1,
                        prior_failure_types=prior,
                        succeeded=False,
                        failure_type=runner.TASK_TIMEOUT_FAILURE,
                        failure_reason="Task timed out after 10 seconds.",
                    )
                    failed["steps"] = [{"step_index": 1, "action": "partial_work"}]
                    return failed
                return _result_payload(
                    config,
                    attempt_number=2,
                    prior_failure_types=prior,
                    succeeded=True,
                )

            with patch.object(
                runner, "_run_single_task_with_timeout", side_effect=run_attempt
            ) as mocked:
                artifact = runner.run_single_task(
                    task_id=TASK_ID,
                    config=config,
                    run_output_dir=config.run.output_dir,
                )

            self.assertTrue(artifact.succeeded)
            self.assertEqual(mocked.call_count, 2)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["attempt_count"], 2)
            self.assertEqual(trace["attempts"][0]["failure_type"], runner.TASK_TIMEOUT_FAILURE)
            self.assertEqual(len(trace["attempts"][0]["steps"]), 1)
            self.assertEqual(trace["attempts"][1]["steps"], [])

    def test_uncaught_exception_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=0)
            success = _result_payload(
                config,
                attempt_number=2,
                prior_failure_types=(runner.TASK_CRASH_FAILURE,),
                succeeded=True,
            )

            with patch.object(
                runner,
                "_run_single_task_core",
                side_effect=[RuntimeError("boom"), success],
            ) as mocked:
                artifact = runner.run_single_task(
                    task_id=TASK_ID,
                    config=config,
                    run_output_dir=config.run.output_dir,
                    model=object(),
                )

            self.assertTrue(artifact.succeeded)
            self.assertEqual(mocked.call_count, 2)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["attempts"][0]["failure_type"], runner.TASK_CRASH_FAILURE)

    def test_max_steps_without_answer_retries_with_clean_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=0, max_steps=1)
            non_terminal = json.dumps(
                {
                    "thought": "Inspect without submitting.",
                    "action": "list_context",
                    "action_input": {},
                }
            )
            valid_answer = json.dumps(
                {
                    "thought": "Submit a clean second-attempt answer.",
                    "action": "answer",
                    "action_input": {"columns": ["value"], "rows": [[1]]},
                }
            )
            model = ScriptedModelAdapter([non_terminal] * 4 + [valid_answer])

            artifact = runner.run_single_task(
                task_id=TASK_ID,
                config=config,
                run_output_dir=config.run.output_dir,
                model=model,
            )

            self.assertTrue(artifact.succeeded)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["attempt_count"], 2)
            self.assertEqual(
                trace["attempts"][0]["failure_type"],
                runner.MAX_STEPS_NO_ANSWER_FAILURE,
            )
            self.assertGreater(len(trace["attempts"][0]["steps"]), 1)
            self.assertEqual(len(trace["attempts"][1]["steps"]), 1)
            self.assertEqual(trace["attempts"][1]["steps"][0]["step_index"], 1)
            self.assertEqual(trace["harness"]["attempt_number"], 2)

    def test_answer_validation_failure_does_not_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=0, max_steps=1)
            invalid_answer = json.dumps(
                {
                    "thought": "Submit an empty answer.",
                    "action": "answer",
                    "action_input": {"columns": ["value"], "rows": []},
                }
            )
            model = ScriptedModelAdapter([invalid_answer] * 4)

            artifact = runner.run_single_task(
                task_id=TASK_ID,
                config=config,
                run_output_dir=config.run.output_dir,
                model=model,
            )

            self.assertFalse(artifact.succeeded)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["attempt_count"], 1)
            self.assertEqual(trace["failure_type"], ANSWER_VALIDATION_FAILED)
            self.assertEqual(trace["attempts"][0]["failure_type"], ANSWER_VALIDATION_FAILED)

    def test_stale_prediction_is_removed_before_each_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root)
            prediction_path = config.run.output_dir / TASK_ID / "prediction.csv"
            prediction_path.parent.mkdir(parents=True)
            prediction_path.write_text("stale\nanswer\n")
            prediction_existed_during_attempt: list[bool] = []

            def run_attempt(**kwargs):
                prediction_existed_during_attempt.append(prediction_path.exists())
                attempt_number = kwargs["attempt_number"]
                prior = kwargs["prior_failure_types"]
                if attempt_number == 1:
                    return _result_payload(
                        config,
                        attempt_number=1,
                        prior_failure_types=prior,
                        succeeded=False,
                        failure_type=runner.TASK_TIMEOUT_FAILURE,
                        failure_reason="Task timed out after 10 seconds.",
                    )
                return _result_payload(
                    config,
                    attempt_number=2,
                    prior_failure_types=prior,
                    succeeded=False,
                    failure_type="model_error",
                    failure_reason="Non-retryable model failure.",
                )

            with patch.object(
                runner, "_run_single_task_with_timeout", side_effect=run_attempt
            ):
                artifact = runner.run_single_task(
                    task_id=TASK_ID,
                    config=config,
                    run_output_dir=config.run.output_dir,
                )

            self.assertEqual(prediction_existed_during_attempt, [False, False])
            self.assertIsNone(artifact.prediction_csv_path)
            self.assertFalse(prediction_path.exists())

    def test_total_time_budget_stops_retries_early(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=10)
            clock = FakeClock()
            allocated_timeouts: list[float] = []

            def run_attempt(**kwargs):
                allocated_timeouts.append(kwargs["timeout_seconds"])
                clock.value += 20 if kwargs["attempt_number"] == 1 else 11
                return _result_payload(
                    config,
                    attempt_number=kwargs["attempt_number"],
                    prior_failure_types=kwargs["prior_failure_types"],
                    succeeded=False,
                    failure_type=runner.TASK_TIMEOUT_FAILURE,
                    failure_reason="Attempt timed out.",
                )

            with (
                patch.object(runner, "monotonic", side_effect=clock.monotonic),
                patch.object(
                    runner, "_run_single_task_with_timeout", side_effect=run_attempt
                ) as mocked,
            ):
                artifact = runner.run_single_task(
                    task_id=TASK_ID,
                    config=config,
                    run_output_dir=config.run.output_dir,
                )

            self.assertFalse(artifact.succeeded)
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(allocated_timeouts, [10.0, 10.0])
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["failure_type"], runner.TASK_TIME_BUDGET_EXHAUSTED_FAILURE)
            self.assertEqual(trace["attempt_count"], 2)

    def test_trace_records_attempt_numbers_and_prior_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root)

            def run_attempt(**kwargs):
                attempt_number = kwargs["attempt_number"]
                prior = kwargs["prior_failure_types"]
                if attempt_number == 1:
                    return _result_payload(
                        config,
                        attempt_number=1,
                        prior_failure_types=prior,
                        succeeded=False,
                        failure_type=runner.TASK_TIMEOUT_FAILURE,
                        failure_reason="Timed out.",
                    )
                if attempt_number == 2:
                    return _result_payload(
                        config,
                        attempt_number=2,
                        prior_failure_types=prior,
                        succeeded=False,
                        failure_reason="Agent did not submit an answer within max_steps.",
                    )
                return _result_payload(
                    config,
                    attempt_number=3,
                    prior_failure_types=prior,
                    succeeded=True,
                )

            with patch.object(
                runner, "_run_single_task_with_timeout", side_effect=run_attempt
            ):
                artifact = runner.run_single_task(
                    task_id=TASK_ID,
                    config=config,
                    run_output_dir=config.run.output_dir,
                )

            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["attempt_number"], 3)
            self.assertEqual(trace["attempt_count"], 3)
            self.assertEqual(
                trace["prior_failure_types"],
                [runner.TASK_TIMEOUT_FAILURE, runner.MAX_STEPS_NO_ANSWER_FAILURE],
            )
            for index, attempt in enumerate(trace["attempts"], start=1):
                self.assertEqual(attempt["attempt_number"], index)
                self.assertEqual(attempt["harness"]["attempt_number"], index)
            self.assertEqual(
                trace["attempts"][2]["harness"]["prior_failure_types"],
                [runner.TASK_TIMEOUT_FAILURE, runner.MAX_STEPS_NO_ANSWER_FAILURE],
            )
            started_event = trace["attempts"][2]["harness"]["events"][0]
            self.assertEqual(started_event["payload"]["attempt_number"], 3)

    def test_successful_task_uses_one_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, timeout_seconds=0)
            answer = json.dumps(
                {
                    "thought": "Submit immediately.",
                    "action": "answer",
                    "action_input": {"columns": ["value"], "rows": [[1]]},
                }
            )

            artifact = runner.run_single_task(
                task_id=TASK_ID,
                config=config,
                run_output_dir=config.run.output_dir,
                model=ScriptedModelAdapter([answer]),
            )

            self.assertTrue(artifact.succeeded)
            trace = json.loads(artifact.trace_path.read_text())
            self.assertEqual(trace["attempt_number"], 1)
            self.assertEqual(trace["attempt_count"], 1)
            self.assertEqual(len(trace["attempts"]), 1)
            self.assertEqual(trace["harness"]["attempt_number"], 1)
            self.assertEqual(trace["prior_failure_types"], [])


if __name__ == "__main__":
    unittest.main()
