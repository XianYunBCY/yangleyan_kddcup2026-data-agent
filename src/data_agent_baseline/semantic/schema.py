from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    name: str
    kind: str
    path: str
    alias: str
    relation: str | None
    primary_key: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DimensionDefinition:
    name: str
    source: str
    expression: str
    data_type: str
    output_label: str
    description: str


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    sources: tuple[str, ...]
    expression: str
    data_type: str
    output_label: str
    description: str


@dataclass(frozen=True, slots=True)
class JoinDefinition:
    left: str
    right: str
    condition: str
    relationship: str


@dataclass(frozen=True, slots=True)
class SemanticModel:
    model_id: str
    version: str
    description: str
    required_paths: tuple[str, ...]
    sources: dict[str, SourceDefinition]
    dimensions: dict[str, DimensionDefinition]
    metrics: dict[str, MetricDefinition]
    joins: tuple[JoinDefinition, ...]

    def public_catalog(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "description": self.description,
            "dimensions": [
                {
                    "name": item.name,
                    "data_type": item.data_type,
                    "output_label": item.output_label,
                    "description": item.description,
                }
                for item in self.dimensions.values()
            ],
            "metrics": [
                {
                    "name": item.name,
                    "data_type": item.data_type,
                    "description": item.description,
                    "output_label": item.output_label,
                }
                for item in self.metrics.values()
            ],
            "filter_operators": ["=", "!=", ">", ">=", "<", "<=", "in", "between"],
        }


@dataclass(frozen=True, slots=True)
class AnalysisSemanticQuery:
    version: str
    dimensions: tuple[str, ...]
    metrics: tuple[str, ...]
    filters: tuple[dict[str, Any], ...]
    group_by: tuple[str, ...]
    order_by: tuple[dict[str, Any], ...]
    output: tuple[str, ...]
    limit: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalysisSemanticQuery:
        return cls(
            version=str(payload["version"]),
            dimensions=tuple(str(item) for item in payload.get("dimensions", [])),
            metrics=tuple(str(item) for item in payload.get("metrics", [])),
            filters=tuple(dict(item) for item in payload.get("filters", [])),
            group_by=tuple(str(item) for item in payload.get("group_by", [])),
            order_by=tuple(dict(item) for item in payload.get("order_by", [])),
            output=tuple(str(item) for item in payload.get("output", [])),
            limit=int(payload["limit"]) if payload.get("limit") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class CompiledSemanticPlan:
    asq: AnalysisSemanticQuery
    semantic_model_id: str
    semantic_model_version: str
    engine: str
    sql: str
    sources: tuple[SourceDefinition, ...]
    join_path: tuple[JoinDefinition, ...]
    output_columns: tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CompiledSemanticPlan:
        return cls(
            asq=AnalysisSemanticQuery.from_dict(dict(payload["asq"])),
            semantic_model_id=str(payload["semantic_model_id"]),
            semantic_model_version=str(payload["semantic_model_version"]),
            engine=str(payload["engine"]),
            sql=str(payload["sql"]),
            sources=tuple(SourceDefinition(**dict(item)) for item in payload.get("sources", [])),
            join_path=tuple(JoinDefinition(**dict(item)) for item in payload.get("join_path", [])),
            output_columns=tuple(str(item) for item in payload.get("output_columns", [])),
            fingerprint=str(payload["fingerprint"]),
        )
