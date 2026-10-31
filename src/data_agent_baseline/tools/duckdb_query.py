from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb

MAX_CELL_DISPLAY_CHARS = 1000


def _compact_cell_for_observation(value: object) -> tuple[object, bool]:
    if isinstance(value, (dict, list, tuple)):
        text = repr(value)
        if len(text) > MAX_CELL_DISPLAY_CHARS:
            type_name = type(value).__name__
            return (
                f"<large nested {type_name} omitted; project scalar fields or expand JSON records with "
                "SELECT (rec).field FROM (SELECT UNNEST(T1.records) AS rec FROM read_json_auto(...) AS T1)>",
                True,
            )
        return value, False

    if isinstance(value, str) and len(value) > MAX_CELL_DISPLAY_CHARS:
        return value[:MAX_CELL_DISPLAY_CHARS] + "... <truncated>", True

    return value, False


def _compact_rows_for_observation(rows: list[object]) -> tuple[list[list[object]], bool]:
    compacted_rows: list[list[object]] = []
    compacted = False
    for row in rows:
        compacted_row: list[object] = []
        for cell in row:
            compacted_cell, cell_compacted = _compact_cell_for_observation(cell)
            compacted_row.append(compacted_cell)
            compacted = compacted or cell_compacted
        compacted_rows.append(compacted_row)
    return compacted_rows, compacted


def _split_select_expressions(select_list: str) -> list[str]:
    expressions: list[str] = []
    start = 0
    depth = 0
    in_string: str | None = None
    escaped = False
    for index, char in enumerate(select_list):
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            expressions.append(select_list[start:index].strip())
            start = index + 1
    expressions.append(select_list[start:].strip())
    return expressions


def _strip_top_level_alias(expression: str) -> str:
    depth = 0
    in_string: str | None = None
    escaped = False
    index = 0
    while index < len(expression):
        char = expression[index]
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            index += 1
            continue
        if char in {"'", '"'}:
            in_string = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and expression[index : index + 2].lower() == "as":
            before = expression[index - 1] if index > 0 else " "
            after = expression[index + 2] if index + 2 < len(expression) else " "
            if before.isspace() and after.isspace():
                return expression[:index].strip()
        index += 1
    return expression.strip()


def _extract_select_expressions(sql: str) -> list[str] | None:
    select_match = re.search(r"\bselect\b", sql, flags=re.IGNORECASE)
    if select_match is None:
        return None
    start = select_match.end()
    depth = 0
    in_string: str | None = None
    escaped = False
    end: int | None = None
    index = start
    while index < len(sql):
        char = sql[index]
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            index += 1
            continue
        if char in {"'", '"'}:
            in_string = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and sql[index : index + 4].lower() == "from":
            before = sql[index - 1] if index > 0 else " "
            after = sql[index + 4] if index + 4 < len(sql) else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                end = index
                break
        index += 1
    if end is None:
        return None
    select_list = sql[start:end].strip()
    if select_list.lower().startswith("distinct "):
        select_list = select_list[9:].strip()
    expressions = _split_select_expressions(select_list)
    if not expressions or any(expression == "*" for expression in expressions):
        return None
    return [_strip_top_level_alias(expression) for expression in expressions]


def _prefer_computed_expression_names(sql: str, column_names: list[str]) -> list[str]:
    expressions = _extract_select_expressions(sql)
    if expressions is None or len(expressions) != len(column_names):
        return column_names

    preferred = list(column_names)
    for index, expression in enumerate(expressions):
        upper_expression = expression.upper()
        is_computed = any(token in upper_expression for token in ("(", "+", "-", "*", "/", "CASE "))
        if is_computed:
            preferred[index] = expression
    return preferred


def execute_duckdb_sql(context_root: Path, sql: str, *, limit: int = 200) -> dict[str, object]:
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "describe", "summarize")):
        raise ValueError("Only read-only DuckDB statements are allowed.")

    original_cwd = Path.cwd()
    os.chdir(context_root)
    try:
        with duckdb.connect(database=":memory:") as conn:
            cursor = conn.execute(sql)
            column_names = [item[0] for item in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)
    finally:
        os.chdir(original_cwd)

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    compacted_rows, cell_truncated = _compact_rows_for_observation(limited_rows)
    result: dict[str, object] = {
        "columns": _prefer_computed_expression_names(sql, column_names),
        "rows": compacted_rows,
        "row_count": len(compacted_rows),
        "truncated": truncated,
    }
    if cell_truncated:
        result["cell_truncated"] = True
        result["note"] = (
            "One or more large nested/long cells were omitted from the observation. "
            "Project scalar fields, or expand JSON records with "
            "`SELECT (rec).field FROM (SELECT UNNEST(T1.records) AS rec FROM read_json_auto(...) AS T1)` "
            "before using the result as an answer."
        )
    return {
        **result,
    }
