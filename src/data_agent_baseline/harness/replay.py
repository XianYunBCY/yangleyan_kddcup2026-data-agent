from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.harness.policy import HarnessPolicy, PREPARE_ASQ_ACTION
from data_agent_baseline.harness.session import HarnessSession, HarnessVersions
from data_agent_baseline.harness.tools import HarnessToolRegistry
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


SQL_ACTIONS = {
    "execute_context_sql",
    "answer_context_sql",
    "execute_duckdb",
    "answer_duckdb",
}


def _load_trace(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Replay trace must contain one JSON object.")
    return payload


def _trace_sha256(trace_path: Path) -> str:
    return hashlib.sha256(trace_path.read_bytes()).hexdigest()


def _extract_asq_plan(trace: dict[str, Any]) -> dict[str, Any]:
    harness = trace.get("harness")
    if not isinstance(harness, dict):
        raise ValueError("Trace does not contain harness data.")
    events = harness.get("events")
    if not isinstance(events, list):
        raise ValueError("Trace harness does not contain events.")
    for event in reversed(events):
        if not isinstance(event, dict) or event.get("event_type") not in {
            "asq.plan.prepared",
            "asq.plan.compiled",
        }:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("plan"), dict):
            return dict(payload["plan"])
    raise ValueError("Trace does not contain a prepared ASQ plan.")


def _extract_sql_step(trace: dict[str, Any], step_index: int | None) -> tuple[str, dict[str, Any]]:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Trace does not contain agent steps.")
    candidates: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step_index is not None and step.get("step_index") != step_index:
            continue
        if step.get("action") in SQL_ACTIONS and isinstance(step.get("action_input"), dict):
            candidates.append(step)
    if not candidates:
        qualifier = f" at step {step_index}" if step_index is not None else ""
        raise ValueError(f"Trace does not contain a replayable SQL action{qualifier}.")
    selected = candidates[-1]
    action = str(selected["action"])
    terminal_action = {
        "execute_context_sql": "answer_context_sql",
        "execute_duckdb": "answer_duckdb",
    }.get(action, action)
    return terminal_action, dict(selected["action_input"])


def replay_trace(
    *,
    task: PublicTask,
    trace_path: Path,
    kind: str,
    step_index: int | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    if kind not in {"asq", "sql"}:
        raise ValueError("Replay kind must be 'asq' or 'sql'.")
    trace = _load_trace(trace_path)
    source_hash = _trace_sha256(trace_path)
    source_harness = trace.get("harness")
    source_versions = source_harness.get("versions", {}) if isinstance(source_harness, dict) else {}
    session = HarnessSession(
        task_id=task.task_id,
        mode="semantic" if kind == "asq" else "raw",
        model="replay",
        max_steps=0,
        versions=HarnessVersions(
            prompt="replay-v1",
            semantic_model=str(source_versions.get("semantic_model", "unversioned")),
        ),
    )
    session.record(
        "replay.started",
        {"kind": kind, "source_trace": str(trace_path), "source_trace_sha256": source_hash},
    )
    registry = HarnessToolRegistry(
        tools or create_default_tool_registry(),
        session,
        HarnessPolicy(session.mode),
    )
    replay_steps: list[dict[str, Any]] = []
    started_at = perf_counter()
    try:
        if kind == "asq":
            plan = _extract_asq_plan(trace)
            is_compiler_plan = plan.get("engine") == "semantic_duckdb"
            if is_compiler_plan:
                prepare_input = {"asq": plan.get("asq")}
            else:
                prepare_input = {
                    "asq": plan.get("asq"),
                    "compiled": {
                        "engine": plan.get("engine"),
                        "path": plan.get("path"),
                        "sql": plan.get("sql"),
                    },
                }
            prepared = registry.execute(task, PREPARE_ASQ_ACTION, prepare_input)
            if is_compiler_plan:
                recorded_version = str(plan.get("semantic_model_version"))
                replay_version = str(prepared.content.get("semantic_model_version"))
                if replay_version != recorded_version:
                    raise ValueError(
                        "ASQ replay semantic model version mismatch: "
                        f"recorded={recorded_version}, current={replay_version}."
                    )
            replay_steps.append(
                {
                    "step_index": 1,
                    "thought": "Replay the recorded ASQ plan.",
                    "action": PREPARE_ASQ_ACTION,
                    "action_input": prepare_input,
                    "raw_response": "__asq_replay__",
                    "observation": {"ok": prepared.ok, "content": prepared.content},
                    "ok": prepared.ok,
                }
            )
            engine = str(plan.get("engine"))
            if engine == "semantic_duckdb":
                action = "answer_prepared_asq"
                action_input = {"fingerprint": str(prepared.content.get("fingerprint"))}
            else:
                action = "answer_context_sql" if engine == "sqlite" else "answer_duckdb"
                action_input = {"sql": str(plan.get("sql"))}
                if engine == "sqlite":
                    action_input["path"] = str(plan.get("path"))
        else:
            try:
                action, action_input = _extract_sql_step(trace, step_index)
            except ValueError:
                plan = _extract_asq_plan(trace)
                if plan.get("engine") != "semantic_duckdb":
                    raise
                recorded_plan = registry.register_recorded_plan(plan)
                action = "answer_prepared_asq"
                action_input = {"fingerprint": recorded_plan.fingerprint}

        result = registry.execute(task, action, action_input)
        replay_steps.append(
            {
                "step_index": len(replay_steps) + 1,
                "thought": f"Replay the recorded {kind.upper()} query without a model call.",
                "action": action,
                "action_input": action_input,
                "raw_response": f"__{kind}_replay__",
                "observation": {"ok": result.ok, "content": result.content},
                "ok": result.ok,
            }
        )
        if not result.is_terminal or result.answer is None:
            raise RuntimeError("Replay query did not produce a terminal answer.")
        session.record(
            "replay.completed",
            {"kind": kind, "source_trace_sha256": source_hash},
            elapsed_ms=(perf_counter() - started_at) * 1000,
        )
        session.finalize(succeeded=True)
        return {
            "task_id": task.task_id,
            "answer": result.answer.to_dict(),
            "steps": replay_steps,
            "failure_reason": None,
            "succeeded": True,
            "replay": {
                "kind": kind,
                "source_trace": str(trace_path),
                "source_trace_sha256": source_hash,
            },
            "harness": session.to_dict(),
        }
    except Exception as exc:
        failure_reason = f"Replay failed: {exc}"
        session.record(
            "replay.failed",
            {"kind": kind, "error_type": type(exc).__name__, "error": str(exc)},
            elapsed_ms=(perf_counter() - started_at) * 1000,
        )
        session.finalize(succeeded=False, failure_reason=failure_reason)
        return {
            "task_id": task.task_id,
            "answer": None,
            "steps": replay_steps,
            "failure_reason": failure_reason,
            "succeeded": False,
            "replay": {
                "kind": kind,
                "source_trace": str(trace_path),
                "source_trace_sha256": source_hash,
            },
            "harness": session.to_dict(),
        }
