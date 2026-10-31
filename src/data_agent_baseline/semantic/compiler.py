from __future__ import annotations

import hashlib
import json
from typing import Any

from data_agent_baseline.semantic.schema import (
    AnalysisSemanticQuery,
    CompiledSemanticPlan,
    DimensionDefinition,
    JoinDefinition,
    SemanticModel,
)


SUPPORTED_OPERATORS = {"=", "!=", ">", ">=", "<", "<=", "in", "between"}


def _string_tuple(asq: dict[str, Any], field: str) -> tuple[str, ...]:
    value = asq.get(field, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"ASQ `{field}` must be a list of semantic names.")
    return tuple(value)


def parse_asq(payload: object) -> AnalysisSemanticQuery:
    if not isinstance(payload, dict):
        raise ValueError("prepare_asq.asq must be an object.")
    if str(payload.get("version", "")) != "1":
        raise ValueError("ASQ version must be '1'.")
    filters = payload.get("filters", [])
    order_by = payload.get("order_by", [])
    if not isinstance(filters, list) or not all(isinstance(item, dict) for item in filters):
        raise ValueError("ASQ `filters` must be a list of objects.")
    if not isinstance(order_by, list) or not all(isinstance(item, dict) for item in order_by):
        raise ValueError("ASQ `order_by` must be a list of objects.")
    output = _string_tuple(payload, "output")
    if not output:
        raise ValueError("ASQ `output` must contain at least one semantic output name.")
    limit = payload.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("ASQ `limit` must be a positive integer when provided.")
    return AnalysisSemanticQuery(
        version="1",
        dimensions=_string_tuple(payload, "dimensions"),
        metrics=_string_tuple(payload, "metrics"),
        filters=tuple(dict(item) for item in filters),
        group_by=_string_tuple(payload, "group_by"),
        order_by=tuple(dict(item) for item in order_by),
        output=output,
        limit=limit,
    )


def _sql_literal(value: Any, data_type: str) -> str:
    if value is None:
        return "NULL"
    if data_type in {"integer", "number"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric filter value, got {value!r}.")
        return str(value)
    if not isinstance(value, str):
        raise ValueError(f"Expected string filter value, got {value!r}.")
    return "'" + value.replace("'", "''") + "'"


def _compile_filter(raw: dict[str, Any], dimensions: dict[str, DimensionDefinition]) -> str:
    field = str(raw.get("field", ""))
    dimension = dimensions.get(field)
    if dimension is None:
        raise ValueError(f"Unknown or non-filterable semantic field: {field}")
    operator = str(raw.get("op", "")).lower()
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported filter operator `{operator}` for `{field}`.")
    value = raw.get("value")
    if value is None and operator in {"=", "!="}:
        return f"{dimension.expression} IS {'NOT ' if operator == '!=' else ''}NULL"
    if operator == "in":
        if not isinstance(value, list) or not value:
            raise ValueError(f"Filter `{field} in` requires a non-empty value list.")
        rendered = ", ".join(_sql_literal(item, dimension.data_type) for item in value)
        return f"{dimension.expression} IN ({rendered})"
    if operator == "between":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Filter `{field} between` requires exactly two values.")
        lower = _sql_literal(value[0], dimension.data_type)
        upper = _sql_literal(value[1], dimension.data_type)
        return f"{dimension.expression} BETWEEN {lower} AND {upper}"
    return f"{dimension.expression} {operator} {_sql_literal(value, dimension.data_type)}"


def _required_sources(asq: AnalysisSemanticQuery, model: SemanticModel) -> set[str]:
    required: set[str] = set()
    for metric_name in asq.metrics:
        metric = model.metrics.get(metric_name)
        if metric is None:
            raise ValueError(f"Unknown semantic metric: {metric_name}")
        required.update(metric.sources)
    dimension_names = list(asq.dimensions) + list(asq.group_by)
    dimension_names.extend(str(item.get("field", "")) for item in asq.filters)
    dimension_names.extend(str(item.get("field", "")) for item in asq.order_by)
    for dimension_name in dimension_names:
        dimension = model.dimensions.get(dimension_name)
        if dimension is None:
            raise ValueError(f"Unknown semantic dimension: {dimension_name}")
        required.add(dimension.source)
    if not required:
        raise ValueError("ASQ must select at least one dimension or metric.")
    return required


def _resolve_join_path(required: set[str], model: SemanticModel) -> tuple[JoinDefinition, ...]:
    ordered_required = [name for name in model.sources if name in required]
    connected = {ordered_required[0]}
    path: list[JoinDefinition] = []
    for target in ordered_required[1:]:
        if target in connected:
            continue
        frontier: list[tuple[str, tuple[JoinDefinition, ...]]] = [
            (source, ()) for source in model.sources if source in connected
        ]
        visited = set(connected)
        selected_path: tuple[JoinDefinition, ...] | None = None
        while frontier:
            current, current_path = frontier.pop(0)
            if current == target:
                selected_path = current_path
                break
            for join in model.joins:
                if join.left == current:
                    neighbor = join.right
                elif join.right == current:
                    neighbor = join.left
                else:
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append((neighbor, (*current_path, join)))
        if selected_path is None:
            raise ValueError(
                "No declared join path connects semantic sources: " + ", ".join(sorted(required))
            )
        for join in selected_path:
            if join not in path:
                path.append(join)
            connected.update((join.left, join.right))
    return tuple(path)


def _validate_grain(
    asq: AnalysisSemanticQuery,
    model: SemanticModel,
    join_path: tuple[JoinDefinition, ...],
) -> None:
    missing_group_by = set(asq.dimensions) - set(asq.group_by)
    if asq.metrics and missing_group_by:
        names = ", ".join(sorted(missing_group_by))
        raise ValueError(f"Metric queries must group by every projected dimension: {names}")
    metric_sources = {
        source for metric_name in asq.metrics for source in model.metrics[metric_name].sources
    }
    for join in join_path:
        if join.relationship == "one_to_many" and join.left in metric_sources:
            raise ValueError(
                f"Metric grain may fan out across `{join.left}` -> `{join.right}`; "
                "define a governed distinct metric before compiling this query."
            )
        if join.relationship == "many_to_one" and join.right in metric_sources:
            raise ValueError(
                f"Metric grain may fan out across `{join.right}` -> `{join.left}`; "
                "define a governed distinct metric before compiling this query."
            )


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compile_asq(model: SemanticModel, payload: object) -> CompiledSemanticPlan:
    asq = parse_asq(payload)
    required = _required_sources(asq, model)
    join_path = _resolve_join_path(required, model)
    _validate_grain(asq, model, join_path)

    selected: list[tuple[str, str]] = []
    for name in asq.dimensions:
        dimension = model.dimensions[name]
        selected.append((dimension.expression, dimension.output_label))
    for name in asq.metrics:
        metric = model.metrics[name]
        selected.append((metric.expression, metric.output_label))
    if len(asq.output) != len(selected):
        raise ValueError(
            f"ASQ output contract has {len(asq.output)} column(s), but the query selects "
            f"{len(selected)} semantic element(s)."
        )
    for requested, (_, governed_label) in zip(asq.output, selected, strict=True):
        accepted = {governed_label, governed_label.rsplit(".", 1)[-1]}
        accepted.update(name.rsplit(".", 1)[-1] for name in (*asq.dimensions, *asq.metrics))
        if requested not in accepted:
            raise ValueError(
                f"Output `{requested}` is not governed by the selected semantic elements. "
                f"Expected `{governed_label}` or its semantic name."
            )

    joined_sources = {source_name for join in join_path for source_name in (join.left, join.right)}
    included_sources = required | joined_sources
    ordered_sources = [model.sources[name] for name in model.sources if name in included_sources]
    base_name = next(name for name in model.sources if name in required)
    base = model.sources[base_name]
    from_sql = f"{base.name} AS {base.alias}"
    connected = {base.name}
    for join in join_path:
        added = join.right if join.left in connected else join.left
        source = model.sources[added]
        from_sql += f" INNER JOIN {source.name} AS {source.alias} ON {join.condition}"
        connected.add(added)

    select_sql = ", ".join(
        f'{expression} AS "{label.replace(chr(34), chr(34) * 2)}"' for expression, label in selected
    )
    sql_parts = [f"SELECT {select_sql}", f"FROM {from_sql}"]
    if asq.filters:
        sql_parts.append(
            "WHERE " + " AND ".join(_compile_filter(raw, model.dimensions) for raw in asq.filters)
        )
    if asq.group_by:
        sql_parts.append(
            "GROUP BY " + ", ".join(model.dimensions[name].expression for name in asq.group_by)
        )
    if asq.order_by:
        order_parts: list[str] = []
        for raw in asq.order_by:
            field = str(raw.get("field", ""))
            direction = str(raw.get("direction", "asc")).lower()
            if direction not in {"asc", "desc"}:
                raise ValueError(f"Invalid order direction `{direction}` for `{field}`.")
            if field in model.dimensions:
                expression = model.dimensions[field].expression
            elif field in model.metrics:
                expression = model.metrics[field].expression
            else:
                raise ValueError(f"Unknown order field: {field}")
            order_parts.append(f"{expression} {direction.upper()}")
        sql_parts.append("ORDER BY " + ", ".join(order_parts))
    if asq.limit is not None:
        sql_parts.append(f"LIMIT {asq.limit}")
    sql = "\n".join(sql_parts)

    fingerprint_payload = {
        "semantic_model_id": model.model_id,
        "semantic_model_version": model.version,
        "asq": asq.to_dict(),
        "sql": sql,
        "sources": [source.name for source in ordered_sources],
    }
    return CompiledSemanticPlan(
        asq=asq,
        semantic_model_id=model.model_id,
        semantic_model_version=model.version,
        engine="semantic_duckdb",
        sql=sql,
        sources=tuple(ordered_sources),
        join_path=join_path,
        output_columns=tuple(label for _, label in selected),
        fingerprint=_fingerprint(fingerprint_payload),
    )
