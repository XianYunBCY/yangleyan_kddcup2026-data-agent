from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from data_agent_baseline.config import AppConfig
from data_agent_baseline.run.runner import create_run_id, run_benchmark


def _read_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize_run(run_dir: Path) -> dict[str, Any]:
    trace_paths = sorted(run_dir.glob("task_*/trace.json"))
    totals = {
        "task_count": len(trace_paths),
        "succeeded_task_count": 0,
        "model_calls": 0,
        "model_calls_with_token_usage": 0,
        "tool_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model_elapsed_ms": 0.0,
        "tool_elapsed_ms": 0.0,
        "e2e_elapsed_seconds": 0.0,
    }
    mode: str | None = None
    for trace_path in trace_paths:
        trace = _read_trace(trace_path)
        totals["succeeded_task_count"] += int(bool(trace.get("succeeded")))
        totals["e2e_elapsed_seconds"] += float(trace.get("e2e_elapsed_seconds") or 0)
        harness = trace.get("harness")
        if not isinstance(harness, dict):
            continue
        mode = mode or str(harness.get("mode"))
        harness_totals = harness.get("totals")
        if not isinstance(harness_totals, dict):
            continue
        for key in (
            "model_calls",
            "model_calls_with_token_usage",
            "tool_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "model_elapsed_ms",
            "tool_elapsed_ms",
        ):
            totals[key] += harness_totals.get(key) or 0
    for key in ("model_elapsed_ms", "tool_elapsed_ms", "e2e_elapsed_seconds"):
        totals[key] = round(float(totals[key]), 3)
    return {"run_dir": str(run_dir), "mode": mode, **totals}


def compare_existing_runs(raw_run_dir: Path, semantic_run_dir: Path) -> dict[str, Any]:
    raw = summarize_run(raw_run_dir)
    semantic = summarize_run(semantic_run_dir)
    return {
        "raw": raw,
        "semantic": semantic,
        "delta_semantic_minus_raw": {
            key: semantic[key] - raw[key]
            for key in (
                "succeeded_task_count",
                "model_calls",
                "model_calls_with_token_usage",
                "tool_calls",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "model_elapsed_ms",
                "tool_elapsed_ms",
                "e2e_elapsed_seconds",
            )
        },
    }


def run_mode_comparison(
    config: AppConfig, *, limit: int | None = None
) -> tuple[Path, dict[str, Any]]:
    base_run_id = config.run.run_id or create_run_id()
    raw_config = replace(
        config,
        harness=replace(config.harness, mode="raw"),
        run=replace(config.run, run_id=f"{base_run_id}.raw"),
    )
    semantic_config = replace(
        config,
        harness=replace(config.harness, mode="semantic"),
        run=replace(config.run, run_id=f"{base_run_id}.semantic"),
    )
    raw_run_dir, _ = run_benchmark(config=raw_config, limit=limit)
    semantic_run_dir, _ = run_benchmark(config=semantic_config, limit=limit)
    comparison = compare_existing_runs(raw_run_dir, semantic_run_dir)
    comparison_path = config.run.output_dir / f"{base_run_id}.comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    return comparison_path, comparison
