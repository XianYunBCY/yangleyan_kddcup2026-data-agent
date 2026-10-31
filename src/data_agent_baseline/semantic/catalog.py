from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.semantic.schema import (
    DimensionDefinition,
    JoinDefinition,
    MetricDefinition,
    SemanticModel,
    SourceDefinition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEMANTIC_MODELS_DIR = PROJECT_ROOT / "semantic_models"


@dataclass(frozen=True, slots=True)
class SemanticMatchDecision:
    model: SemanticModel
    missing_paths: tuple[str, ...]
    missing_dimensions: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    missing_filter_fields: tuple[str, ...]
    missing_order_fields: tuple[str, ...]

    @property
    def source_compatible(self) -> bool:
        return not self.missing_paths

    @property
    def matched(self) -> bool:
        return not any(
            (
                self.missing_paths,
                self.missing_dimensions,
                self.missing_metrics,
                self.missing_filter_fields,
                self.missing_order_fields,
            )
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        for label, values in (
            ("missing required paths", self.missing_paths),
            ("missing dimensions", self.missing_dimensions),
            ("missing metrics", self.missing_metrics),
            ("missing filter fields", self.missing_filter_fields),
            ("missing order fields", self.missing_order_fields),
        ):
            if values:
                reasons.append(f"{label}: {', '.join(values)}")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model.model_id,
            "model_version": self.model.version,
            "source_compatible": self.source_compatible,
            "matched": self.matched,
            "missing_paths": list(self.missing_paths),
            "missing_dimensions": list(self.missing_dimensions),
            "missing_metrics": list(self.missing_metrics),
            "missing_filter_fields": list(self.missing_filter_fields),
            "missing_order_fields": list(self.missing_order_fields),
            "reasons": list(self.reasons),
        }


def _required_mapping(payload: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source}: `{key}` must be a mapping.")
    return value


def _load_model(path: Path) -> SemanticModel:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: semantic model must be a mapping.")
    match = _required_mapping(payload, "match", path)
    source_payload = _required_mapping(payload, "sources", path)
    dimension_payload = _required_mapping(payload, "dimensions", path)
    metric_payload = _required_mapping(payload, "metrics", path)
    join_payload = payload.get("joins", [])
    if not isinstance(join_payload, list):
        raise ValueError(f"{path}: `joins` must be a list.")

    sources: dict[str, SourceDefinition] = {}
    for name, raw in source_payload.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: source `{name}` must be a mapping.")
        kind = str(raw.get("kind", ""))
        if kind not in {"sqlite", "csv", "json"}:
            raise ValueError(f"{path}: source `{name}` has unsupported kind `{kind}`.")
        sources[str(name)] = SourceDefinition(
            name=str(name),
            kind=kind,
            path=str(raw["path"]),
            alias=str(raw["alias"]),
            relation=str(raw["relation"]) if raw.get("relation") is not None else None,
            primary_key=tuple(str(item) for item in raw.get("primary_key", [])),
        )

    dimensions: dict[str, DimensionDefinition] = {}
    for name, raw in dimension_payload.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: dimension `{name}` must be a mapping.")
        source_name = str(raw["source"])
        if source_name not in sources:
            raise ValueError(f"{path}: dimension `{name}` uses unknown source `{source_name}`.")
        dimensions[str(name)] = DimensionDefinition(
            name=str(name),
            source=source_name,
            expression=str(raw["expression"]),
            data_type=str(raw["data_type"]),
            output_label=str(raw.get("output_label", str(name).rsplit(".", 1)[-1])),
            description=str(raw.get("description", "")),
        )

    metrics: dict[str, MetricDefinition] = {}
    for name, raw in metric_payload.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: metric `{name}` must be a mapping.")
        metric_sources = tuple(str(item) for item in raw.get("sources", []))
        unknown_sources = set(metric_sources) - set(sources)
        if unknown_sources:
            raise ValueError(f"{path}: metric `{name}` uses unknown sources {unknown_sources}.")
        metrics[str(name)] = MetricDefinition(
            name=str(name),
            sources=metric_sources,
            expression=str(raw["expression"]),
            data_type=str(raw["data_type"]),
            output_label=str(raw["output_label"]),
            description=str(raw.get("description", "")),
        )

    joins: list[JoinDefinition] = []
    for raw in join_payload:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: each join must be a mapping.")
        left = str(raw["left"])
        right = str(raw["right"])
        if left not in sources or right not in sources:
            raise ValueError(f"{path}: join `{left}` -> `{right}` uses an unknown source.")
        relationship = str(raw["relationship"])
        if relationship not in {"one_to_one", "one_to_many", "many_to_one"}:
            raise ValueError(f"{path}: unsupported join relationship `{relationship}`.")
        joins.append(
            JoinDefinition(
                left=left,
                right=right,
                condition=str(raw["condition"]),
                relationship=relationship,
            )
        )

    required_paths = match.get("required_paths", [])
    if not isinstance(required_paths, list) or not required_paths:
        raise ValueError(f"{path}: match.required_paths must be a non-empty list.")
    return SemanticModel(
        model_id=str(payload["id"]),
        version=str(payload["version"]),
        description=str(payload.get("description", "")),
        required_paths=tuple(str(item) for item in required_paths),
        sources=sources,
        dimensions=dimensions,
        metrics=metrics,
        joins=tuple(joins),
    )


class SemanticCatalog:
    def __init__(self, models: tuple[SemanticModel, ...]) -> None:
        self.models = models

    def candidate_models(self, task: PublicTask) -> tuple[SemanticModel, ...]:
        return tuple(
            model
            for model in self.models
            if all((task.context_dir / path).is_file() for path in model.required_paths)
        )

    def evaluate_task(
        self,
        task: PublicTask,
        *,
        required_dimensions: Iterable[str] = (),
        required_metrics: Iterable[str] = (),
        required_filter_fields: Iterable[str] = (),
        required_order_fields: Iterable[str] = (),
    ) -> tuple[SemanticMatchDecision, ...]:
        dimensions = set(required_dimensions)
        metrics = set(required_metrics)
        filter_fields = set(required_filter_fields)
        order_fields = set(required_order_fields)
        decisions: list[SemanticMatchDecision] = []
        for model in self.models:
            missing_paths = tuple(
                path for path in model.required_paths if not (task.context_dir / path).is_file()
            )
            decisions.append(
                SemanticMatchDecision(
                    model=model,
                    missing_paths=missing_paths,
                    missing_dimensions=tuple(sorted(dimensions - set(model.dimensions))),
                    missing_metrics=tuple(sorted(metrics - set(model.metrics))),
                    missing_filter_fields=tuple(sorted(filter_fields - set(model.dimensions))),
                    missing_order_fields=tuple(
                        sorted(order_fields - (set(model.dimensions) | set(model.metrics)))
                    ),
                )
            )
        return tuple(decisions)

    def match_task(
        self,
        task: PublicTask,
        *,
        required_dimensions: Iterable[str],
        required_metrics: Iterable[str],
        required_filter_fields: Iterable[str],
        required_order_fields: Iterable[str] = (),
    ) -> SemanticModel | None:
        decisions = self.evaluate_task(
            task,
            required_dimensions=required_dimensions,
            required_metrics=required_metrics,
            required_filter_fields=required_filter_fields,
            required_order_fields=required_order_fields,
        )
        matches = [decision.model for decision in decisions if decision.matched]
        if len(matches) > 1:
            ids = ", ".join(model.model_id for model in matches)
            raise ValueError(f"Multiple semantic models match {task.task_id}: {ids}")
        return matches[0] if matches else None

    def by_id(self, model_id: str) -> SemanticModel:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(f"Unknown semantic model: {model_id}")


def load_semantic_catalog(models_dir: Path = DEFAULT_SEMANTIC_MODELS_DIR) -> SemanticCatalog:
    if not models_dir.is_dir():
        return SemanticCatalog(())
    models = tuple(_load_model(path) for path in sorted(models_dir.glob("*.yaml")))
    return SemanticCatalog(models)
