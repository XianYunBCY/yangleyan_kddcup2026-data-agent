from __future__ import annotations

import csv
import json
import multiprocessing
import queue as queue_module
import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, perf_counter, sleep
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.prompt import REACT_SYSTEM_PROMPT, SEMANTIC_HARNESS_SYSTEM_PROMPT
from data_agent_baseline.agents.react import (
    ReActAgent,
    ReActAgentConfig,
    is_transient_model_connection_error,
)
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.harness.policy import HarnessPolicy
from data_agent_baseline.harness.session import HarnessSession, HarnessVersions
from data_agent_baseline.harness.telemetry import InstrumentedModelAdapter
from data_agent_baseline.harness.tools import HarnessToolRegistry, PREPARE_ASQ_DESCRIPTION
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


MAX_TASK_RETRIES = 3
MAX_TASK_ATTEMPTS = MAX_TASK_RETRIES + 1
TASK_TIME_BUDGET_MULTIPLIER = 3
MODEL_CONNECTION_RETRY_DELAY_SECONDS = 3.0

TASK_TIMEOUT_FAILURE = "task_timeout"
TASK_CRASH_FAILURE = "task_crash"
MAX_STEPS_NO_ANSWER_FAILURE = "max_steps_no_answer"
TASK_TIME_BUDGET_EXHAUSTED_FAILURE = "task_time_budget_exhausted"
RETRYABLE_TASK_FAILURES = {
    TASK_TIMEOUT_FAILURE,
    TASK_CRASH_FAILURE,
    MAX_STEPS_NO_ANSWER_FAILURE,
}


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path)
            if self.prediction_csv_path
            else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _system_prompt_for_mode(mode: str) -> str:
    return SEMANTIC_HARNESS_SYSTEM_PROMPT if mode == "semantic" else REACT_SYSTEM_PROMPT


def _build_harness_session(
    *,
    task_id: str,
    config: AppConfig,
    tool_descriptions: str,
    attempt_number: int = 1,
    prior_failure_types: tuple[str, ...] = (),
) -> HarnessSession:
    system_prompt = _system_prompt_for_mode(config.harness.mode)
    effective_tool_descriptions = tool_descriptions
    if config.harness.mode == "semantic":
        effective_tool_descriptions = f"{PREPARE_ASQ_DESCRIPTION}\n{tool_descriptions}"
    session = HarnessSession(
        task_id=task_id,
        mode=config.harness.mode,
        model=config.agent.model,
        max_steps=config.agent.max_steps,
        attempt_number=attempt_number,
        prior_failure_types=prior_failure_types,
        versions=HarnessVersions(
            prompt=config.harness.prompt_version,
            prompt_sha256=_sha256_text(system_prompt),
            tools_sha256=_sha256_text(effective_tool_descriptions),
            semantic_model=config.harness.semantic_model_version,
        ),
    )
    session.record(
        "session.started",
        {
            "mode": config.harness.mode,
            "model": config.agent.model,
            "max_steps": config.agent.max_steps,
            "attempt_number": attempt_number,
            "prior_failure_types": list(prior_failure_types),
        },
    )
    return session


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(
    task_id: str,
    failure_reason: str,
    config: AppConfig | None = None,
    *,
    failure_type: str | None = None,
    attempt_number: int = 1,
    prior_failure_types: tuple[str, ...] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "failure_type": failure_type,
        "succeeded": False,
    }
    if config is not None:
        session = _build_harness_session(
            task_id=task_id,
            config=config,
            tool_descriptions="",
            attempt_number=attempt_number,
            prior_failure_types=prior_failure_types,
        )
        session.finalize(succeeded=False, failure_reason=failure_reason)
        payload["harness"] = session.to_dict()
    return payload


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    attempt_number: int = 1,
    prior_failure_types: tuple[str, ...] = (),
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    base_tools = tools or create_default_tool_registry()
    session = _build_harness_session(
        task_id=task_id,
        config=config,
        tool_descriptions=base_tools.describe_for_prompt(),
        attempt_number=attempt_number,
        prior_failure_types=prior_failure_types,
    )
    harness_tools = HarnessToolRegistry(
        base_tools,
        session,
        HarnessPolicy(config.harness.mode),
    )
    instrumented_model = InstrumentedModelAdapter(model or build_model_adapter(config), session)
    agent = ReActAgent(
        model=instrumented_model,
        tools=harness_tools,
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            validate_answer_sanity=config.harness.mode == "raw",
        ),
        system_prompt=_system_prompt_for_mode(config.harness.mode),
    )
    run_result = agent.run(task)
    session.finalize(
        succeeded=run_result.succeeded,
        failure_reason=run_result.effective_failure_reason,
    )
    payload = run_result.to_dict()
    payload["harness"] = session.to_dict()
    return payload


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    queue: multiprocessing.Queue[Any],
    attempt_number: int,
    prior_failure_types: tuple[str, ...],
) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(
                    task_id=task_id,
                    config=config,
                    attempt_number=attempt_number,
                    prior_failure_types=prior_failure_types,
                ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    timeout_seconds: float | None = None,
    attempt_number: int = 1,
    prior_failure_types: tuple[str, ...] = (),
) -> dict[str, Any]:
    effective_timeout = (
        float(config.run.task_timeout_seconds) if timeout_seconds is None else timeout_seconds
    )
    if effective_timeout <= 0:
        return _run_single_task_core(
            task_id=task_id,
            config=config,
            attempt_number=attempt_number,
            prior_failure_types=prior_failure_types,
        )

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue, attempt_number, prior_failure_types),
    )
    process.start()
    deadline = monotonic() + effective_timeout
    result: dict[str, Any] | None = None
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        try:
            result = queue.get(timeout=min(0.25, remaining))
            break
        except queue_module.Empty:
            if not process.is_alive():
                break

    if result is None:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            return _failure_run_result_payload(
                task_id,
                f"Task timed out after {effective_timeout:g} seconds.",
                config,
                failure_type=TASK_TIMEOUT_FAILURE,
                attempt_number=attempt_number,
                prior_failure_types=prior_failure_types,
            )
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
                config,
                failure_type=TASK_CRASH_FAILURE,
                attempt_number=attempt_number,
                prior_failure_types=prior_failure_types,
            )
        return _failure_run_result_payload(
            task_id,
            "Task exited without returning a result.",
            config,
            failure_type=TASK_CRASH_FAILURE,
            attempt_number=attempt_number,
            prior_failure_types=prior_failure_types,
        )

    process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)

    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(
        task_id,
        f"Task failed with uncaught error: {result['error']}",
        config,
        failure_type=TASK_CRASH_FAILURE,
        attempt_number=attempt_number,
        prior_failure_types=prior_failure_types,
    )


def _write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if bool(run_result.get("succeeded")) and isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def write_task_outputs(
    task_id: str,
    run_output_dir: Path,
    run_result: dict[str, Any],
) -> TaskRunArtifacts:
    return _write_task_outputs(task_id, run_output_dir, run_result)


def _is_retryable_model_connection_failure(run_result: dict[str, Any]) -> bool:
    failure_reason = run_result.get("failure_reason")
    return (
        isinstance(failure_reason, str)
        and failure_reason.startswith("__model_error__:")
        and is_transient_model_connection_error(failure_reason)
    )


def _attempt_failure_type(run_result: dict[str, Any]) -> str | None:
    if bool(run_result.get("succeeded")):
        return None
    failure_type = run_result.get("failure_type")
    if isinstance(failure_type, str) and failure_type:
        return failure_type
    if run_result.get("failure_reason") == "Agent did not submit an answer within max_steps.":
        return MAX_STEPS_NO_ANSWER_FAILURE
    if _is_retryable_model_connection_failure(run_result):
        return TASK_CRASH_FAILURE
    return None


def _clear_stale_prediction(run_output_dir: Path, task_id: str) -> None:
    prediction_path = run_output_dir / task_id / "prediction.csv"
    if prediction_path.exists() or prediction_path.is_symlink():
        prediction_path.unlink()


def _mark_time_budget_exhausted(
    run_result: dict[str, Any], *, total_budget_seconds: float
) -> dict[str, Any]:
    payload = dict(run_result)
    payload.update(
        {
            "answer": None,
            "succeeded": False,
            "failure_type": TASK_TIME_BUDGET_EXHAUSTED_FAILURE,
            "failure_reason": (
                "Task retry wall-clock budget exhausted after "
                f"{total_budget_seconds:g} seconds."
            ),
        }
    )
    return payload


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    budget_started_at = monotonic()
    single_timeout_seconds = float(config.run.task_timeout_seconds)
    total_budget_seconds = single_timeout_seconds * TASK_TIME_BUDGET_MULTIPLIER
    budget_deadline = (
        budget_started_at + total_budget_seconds if single_timeout_seconds > 0 else None
    )

    attempts: list[dict[str, Any]] = []
    prior_failure_types: list[str] = []
    run_result: dict[str, Any] | None = None

    for attempt_number in range(1, MAX_TASK_ATTEMPTS + 1):
        _clear_stale_prediction(run_output_dir, task_id)
        remaining_budget = None
        if budget_deadline is not None:
            remaining_budget = budget_deadline - monotonic()
            if remaining_budget <= 0:
                if run_result is None:
                    run_result = _failure_run_result_payload(
                        task_id,
                        (
                            "Task retry wall-clock budget exhausted after "
                            f"{total_budget_seconds:g} seconds."
                        ),
                        config,
                        failure_type=TASK_TIME_BUDGET_EXHAUSTED_FAILURE,
                        attempt_number=attempt_number,
                        prior_failure_types=tuple(prior_failure_types),
                    )
                else:
                    run_result = _mark_time_budget_exhausted(
                        run_result,
                        total_budget_seconds=total_budget_seconds,
                    )
                break

        attempt_started_at = perf_counter()
        prior_failures = tuple(prior_failure_types)
        try:
            if model is None and tools is None:
                attempt_timeout = single_timeout_seconds
                if remaining_budget is not None:
                    attempt_timeout = min(attempt_timeout, remaining_budget)
                attempt_result = _run_single_task_with_timeout(
                    task_id=task_id,
                    config=config,
                    timeout_seconds=attempt_timeout,
                    attempt_number=attempt_number,
                    prior_failure_types=prior_failures,
                )
            else:
                attempt_result = _run_single_task_core(
                    task_id=task_id,
                    config=config,
                    model=model,
                    tools=tools,
                    attempt_number=attempt_number,
                    prior_failure_types=prior_failures,
                )
        except Exception as exc:  # noqa: BLE001
            attempt_result = _failure_run_result_payload(
                task_id,
                f"Task failed with uncaught error: {exc}",
                config,
                failure_type=TASK_CRASH_FAILURE,
                attempt_number=attempt_number,
                prior_failure_types=prior_failures,
            )

        failure_type = _attempt_failure_type(attempt_result)
        attempt_result["attempt_number"] = attempt_number
        attempt_result["prior_failure_types"] = list(prior_failures)
        attempt_result["failure_type"] = failure_type
        attempt_result["attempt_elapsed_seconds"] = round(
            perf_counter() - attempt_started_at, 3
        )
        attempts.append(dict(attempt_result))
        run_result = attempt_result

        if bool(attempt_result.get("succeeded")):
            break
        if failure_type not in RETRYABLE_TASK_FAILURES:
            break
        if attempt_number >= MAX_TASK_ATTEMPTS:
            break

        prior_failure_types.append(failure_type)
        if budget_deadline is not None and monotonic() >= budget_deadline:
            run_result = _mark_time_budget_exhausted(
                run_result,
                total_budget_seconds=total_budget_seconds,
            )
            break
        if _is_retryable_model_connection_failure(attempt_result):
            sleep(MODEL_CONNECTION_RETRY_DELAY_SECONDS)

    if run_result is None:
        raise RuntimeError("Task attempt loop finished without producing a result.")

    run_result["attempt_number"] = len(attempts)
    run_result["attempt_count"] = len(attempts)
    run_result["prior_failure_types"] = list(
        attempts[-1].get("prior_failure_types", []) if attempts else []
    )
    run_result["attempts"] = attempts
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir, run_id=config.run.run_id
    )

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=model,
                tools=tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "harness_mode": config.harness.mode,
            "prompt_version": config.harness.prompt_version,
            "semantic_model_version": config.harness.semantic_model_version,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
