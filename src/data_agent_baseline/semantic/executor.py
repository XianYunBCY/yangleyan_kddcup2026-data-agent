from __future__ import annotations

import json
import sqlite3
from typing import Any

import duckdb
import pandas as pd

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.semantic.schema import CompiledSemanticPlan, SourceDefinition
from data_agent_baseline.tools.duckdb_query import _compact_rows_for_observation
from data_agent_baseline.tools.filesystem import resolve_context_path


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _register_source(
    conn: duckdb.DuckDBPyConnection,
    task: PublicTask,
    source: SourceDefinition,
) -> None:
    path = resolve_context_path(task, source.path)
    if source.kind == "csv":
        conn.read_csv(str(path), header=True, auto_detect=True).create_view(source.name)
        return
    if source.kind == "json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            records = payload.get("records")
            declared_relation = payload.get("table")
            if source.relation and declared_relation != source.relation:
                raise ValueError(
                    f"JSON semantic source `{source.name}` declares table "
                    f"`{declared_relation}`, expected `{source.relation}`."
                )
        else:
            records = payload
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise ValueError(
                f"JSON semantic source `{source.name}` must be a record list or contain "
                "a `records` list."
            )
        conn.register(source.name, pd.DataFrame(records))
        return
    if source.kind == "sqlite":
        if not source.relation:
            raise ValueError(f"SQLite semantic source `{source.name}` is missing a relation name.")
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as sqlite_conn:
            cursor = sqlite_conn.execute(
                f"SELECT * FROM {_quote_sqlite_identifier(source.relation)}"
            )
            columns = [item[0] for item in cursor.description or []]
            frame = pd.DataFrame(cursor.fetchall(), columns=columns)
        conn.register(source.name, frame)
        return
    raise ValueError(f"Unsupported semantic source kind: {source.kind}")


def execute_semantic_plan(
    task: PublicTask,
    plan: CompiledSemanticPlan,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    with duckdb.connect(database=":memory:") as conn:
        for source in plan.sources:
            _register_source(conn, task, source)
        cursor = conn.execute(plan.sql)
        columns = [str(item[0]) for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)
    truncated = len(rows) > limit
    compacted_rows, cell_truncated = _compact_rows_for_observation(rows[:limit])
    result: dict[str, Any] = {
        "columns": columns,
        "rows": compacted_rows,
        "row_count": len(compacted_rows),
        "truncated": truncated,
        "semantic_model_id": plan.semantic_model_id,
        "semantic_model_version": plan.semantic_model_version,
        "plan_fingerprint": plan.fingerprint,
    }
    if cell_truncated:
        result["cell_truncated"] = True
    return result
