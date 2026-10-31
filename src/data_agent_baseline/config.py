from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    mode: str = "raw"
    prompt_version: str = "react-v1"
    semantic_model_version: str = "unversioned"


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _string_value(raw_value: object, default_value: str) -> str:
    if raw_value is None:
        return default_value
    text = str(raw_value)

    def replace_match(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(2)
        value = os.environ.get(name)
        if value is not None:
            return value
        if fallback is not None:
            return fallback
        return ""

    return ENV_PATTERN.sub(replace_match, text)


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    harness_defaults = HarnessConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    harness_payload = payload.get("harness", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=_string_value(agent_payload.get("model"), agent_defaults.model),
        api_base=_string_value(agent_payload.get("api_base"), agent_defaults.api_base),
        api_key=_string_value(agent_payload.get("api_key"), agent_defaults.api_key),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
    )
    harness_config = HarnessConfig(
        mode=_string_value(harness_payload.get("mode"), harness_defaults.mode).strip().lower(),
        prompt_version=_string_value(
            harness_payload.get("prompt_version"), harness_defaults.prompt_version
        ),
        semantic_model_version=_string_value(
            harness_payload.get("semantic_model_version"),
            harness_defaults.semantic_model_version,
        ),
    )
    if harness_config.mode not in {"raw", "semantic"}:
        raise ValueError("harness.mode must be 'raw' or 'semantic'.")
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(
            run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)
        ),
    )
    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        harness=harness_config,
        run=run_config,
    )
