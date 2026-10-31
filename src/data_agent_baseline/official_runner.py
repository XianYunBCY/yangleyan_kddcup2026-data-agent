from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    HarnessConfig,
    RunConfig,
)
from data_agent_baseline.run.runner import TaskRunArtifacts, run_single_task


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


def _build_config(input_root: Path, log_root: Path) -> AppConfig:
    api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    api_base = (
        os.environ.get("MODEL_API_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    model = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"

    return AppConfig(
        dataset=DatasetConfig(root_path=input_root),
        agent=AgentConfig(
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_steps=_env_int("DABENCH_MAX_STEPS", 16),
            temperature=_env_float("DABENCH_TEMPERATURE", 0.0),
        ),
        harness=HarnessConfig(
            mode=os.environ.get("DABENCH_HARNESS_MODE", "raw").strip().lower(),
            prompt_version=os.environ.get("DABENCH_PROMPT_VERSION", "react-v1"),
            semantic_model_version=os.environ.get("DABENCH_SEMANTIC_MODEL_VERSION", "unversioned"),
        ),
        run=RunConfig(
            output_dir=log_root / "artifacts",
            run_id=None,
            max_workers=_env_int("DABENCH_MAX_WORKERS", 2),
            task_timeout_seconds=_env_int("DABENCH_TASK_TIMEOUT_SECONDS", 300),
        ),
    )


def _configure_logging(log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=os.environ.get("DABENCH_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_root / "runtime.log", encoding="utf-8"),
        ],
    )


def _publish_prediction(artifact: TaskRunArtifacts, output_root: Path) -> Path | None:
    if artifact.prediction_csv_path is None:
        return None
    task_output_dir = output_root / artifact.task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    target = task_output_dir / "prediction.csv"
    shutil.copyfile(artifact.prediction_csv_path, target)
    return target


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def main() -> None:
    input_root = Path(os.environ.get("INPUT_ROOT", "/input")).resolve()
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/output")).resolve()
    log_root = Path(os.environ.get("LOG_ROOT", "/logs")).resolve()

    _configure_logging(log_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _build_config(input_root, log_root)
    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    task_ids = [task.task_id for task in tasks]
    max_workers = max(1, config.run.max_workers)

    logging.info("Input root: %s", input_root)
    logging.info("Output root: %s", output_root)
    logging.info("Log root: %s", log_root)
    logging.info("Model: %s", config.agent.model)
    logging.info("API base: %s", config.agent.api_base)
    logging.info("Tasks: %d, max_workers: %d", len(task_ids), max_workers)

    artifacts_root = config.run.output_dir / "official"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    started_at = perf_counter()
    artifacts: list[TaskRunArtifacts] = []

    if not task_ids:
        logging.warning("No task directories found under %s", input_root)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task_id = {
            executor.submit(
                run_single_task,
                task_id=task_id,
                config=config,
                run_output_dir=artifacts_root,
            ): task_id
            for task_id in task_ids
        }
        for future in as_completed(future_to_task_id):
            task_id = future_to_task_id[future]
            try:
                artifact = future.result()
            except BaseException as exc:  # noqa: BLE001
                logging.exception("Task %s crashed before artifact generation: %s", task_id, exc)
                continue

            artifacts.append(artifact)
            published_path = _publish_prediction(artifact, output_root)
            if artifact.succeeded:
                logging.info("Task %s succeeded; prediction=%s", task_id, published_path)
            else:
                logging.warning("Task %s failed: %s", task_id, artifact.failure_reason)

    artifacts.sort(key=lambda artifact: int(artifact.task_id.removeprefix("task_")))
    summary = {
        "task_count": len(task_ids),
        "attempted_task_count": len(artifacts),
        "succeeded_task_count": sum(1 for artifact in artifacts if artifact.succeeded),
        "elapsed_seconds": round(perf_counter() - started_at, 3),
        "max_workers": max_workers,
        "tasks": [artifact.to_dict() for artifact in artifacts],
        "config": {
            "dataset": {"root_path": str(config.dataset.root_path)},
            "agent": {
                **asdict(config.agent),
                "api_key": "<redacted>" if config.agent.api_key else "",
            },
            "harness": asdict(config.harness),
            "run": asdict(config.run),
        },
    }
    _write_json(log_root / "summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    logging.info(
        "Completed %d/%d tasks in %.3fs",
        summary["succeeded_task_count"],
        len(task_ids),
        summary["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
