from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    extract_document_records,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
    search_context_text,
)
from data_agent_baseline.tools.duckdb_query import execute_duckdb_sql
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30
TOOL_EXECUTION_TIMEOUT_SECONDS = 45
MAX_PREVIEW_ROWS = 200
MAX_FINAL_ROWS = 5000
MAX_PREVIEW_CHARS = 8000


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = min(int(action_input.get("max_rows", 20)), MAX_PREVIEW_ROWS)
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = min(int(action_input.get("max_chars", 4000)), MAX_PREVIEW_CHARS)
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = min(int(action_input.get("max_chars", 4000)), MAX_PREVIEW_CHARS)
    start = int(action_input.get("start", 0))
    return ToolExecutionResult(
        ok=True, content=read_doc_preview(task, path, max_chars=max_chars, start=start)
    )


def _search_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    query = str(action_input["query"])
    path = action_input.get("path")
    content = search_context_text(
        task,
        query,
        path=str(path) if path else None,
        regex=bool(action_input.get("regex", False)),
        case_sensitive=bool(action_input.get("case_sensitive", False)),
        max_matches=int(action_input.get("max_matches", 20)),
        context_chars=int(action_input.get("context_chars", 120)),
    )
    return ToolExecutionResult(ok=True, content=content)


def _augment_doc_record_fields(
    task: PublicTask, path: str, fields: list[str] | None
) -> list[str] | None:
    del task
    del path
    if fields is not None:
        return list(dict.fromkeys(fields))
    return None


def _extract_doc_records(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    raw_fields = action_input.get("fields")
    fields = [str(field) for field in raw_fields] if isinstance(raw_fields, list) else None
    fields = _augment_doc_record_fields(task, path, fields)
    return ToolExecutionResult(
        ok=True,
        content=extract_document_records(
            task,
            path,
            fields=fields,
            max_records=int(action_input.get("max_records", 1000)),
            include_partial=bool(action_input.get("include_partial", False)),
        ),
    )


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = min(int(action_input.get("limit", 200)), MAX_PREVIEW_ROWS)
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _answer_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = min(int(action_input.get("limit", MAX_FINAL_ROWS)), MAX_FINAL_ROWS)
    content = execute_read_only_sql(path, sql, limit=limit)
    if content.get("truncated") is True:
        raise ValueError(
            "Final SQL result was truncated. Refine the query so the final answer is smaller."
        )
    columns = content.get("columns")
    rows = content.get("rows")
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError("Final SQL query did not produce valid column names.")
    if not isinstance(rows, list):
        raise ValueError("Final SQL query did not produce valid rows.")
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
            "source": "answer_context_sql",
        },
        is_terminal=True,
        answer=AnswerTable(columns=columns, rows=rows),
    )


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _execute_duckdb(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    limit = min(int(action_input.get("limit", 200)), MAX_PREVIEW_ROWS)
    return ToolExecutionResult(
        ok=True,
        content=execute_duckdb_sql(task.context_dir, sql, limit=limit),
    )


def _answer_duckdb(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    limit = min(int(action_input.get("limit", MAX_FINAL_ROWS)), MAX_FINAL_ROWS)
    content = execute_duckdb_sql(task.context_dir, sql, limit=limit)
    if content.get("truncated") is True:
        raise ValueError(
            "Final DuckDB result was truncated. Refine the query so the final answer is smaller."
        )
    if content.get("cell_truncated") is True:
        raise ValueError(
            "Final DuckDB result contains large nested or truncated cells. Project scalar fields "
            "or expand nested JSON records before submitting the answer."
        )
    columns = content.get("columns")
    rows = content.get("rows")
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError("Final DuckDB query did not produce valid column names.")
    if not isinstance(rows, list):
        raise ValueError("Final DuckDB query did not produce valid rows.")
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
            "source": "answer_duckdb",
        },
        is_terminal=True,
        answer=AnswerTable(columns=columns, rows=rows),
    )


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in columns)
    ):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        handler = self.handlers[action]
        if action == "answer":
            return handler(task, action_input)

        result_queue: queue.Queue[tuple[str, ToolExecutionResult | Exception]] = queue.Queue(
            maxsize=1
        )

        def _execute_in_daemon_thread() -> None:
            try:
                result_queue.put(("ok", handler(task, action_input)))
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("error", exc))

        thread = threading.Thread(target=_execute_in_daemon_thread, daemon=True)
        thread.start()
        thread.join(timeout=TOOL_EXECUTION_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise TimeoutError(
                f"Tool `{action}` exceeded hard timeout of {TOOL_EXECUTION_TIMEOUT_SECONDS} seconds."
            )
        status, payload = result_queue.get_nowait()
        if status == "ok" and isinstance(payload, ToolExecutionResult):
            return payload
        if isinstance(payload, Exception):
            raise payload
        raise RuntimeError(f"Tool `{action}` failed without a result.")


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description=(
                "Submit the final answer table when the result did not come directly from "
                "a final SQL query."
            ),
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={
                "path": "relative/path/to/file.sqlite",
                "sql": "SELECT ...",
                "limit": 200,
            },
        ),
        "answer_context_sql": ToolSpec(
            name="answer_context_sql",
            description=(
                "Submit the final answer by running one read-only SQL query against a "
                "sqlite/db file inside context. Use this instead of execute_context_sql "
                "when the SQL result is the final answer table."
            ),
            input_schema={
                "path": "relative/path/to/file.sqlite",
                "sql": "SELECT ...",
                "limit": 200,
            },
        ),
        "execute_duckdb": ToolSpec(
            name="execute_duckdb",
            description=(
                "Run read-only DuckDB SQL in the task context directory. Use this for CSV/JSON "
                "aggregation and joins, e.g. read_csv_auto('csv/file.csv') or "
                "read_json_auto('json/file.json'). Do not use this to read SQLite .db files; "
                "use execute_context_sql for SQLite-only queries or execute_python with sqlite3 "
                "when a SQLite table must be joined to CSV/JSON files."
            ),
            input_schema={"sql": "SELECT ...", "limit": 200},
        ),
        "answer_duckdb": ToolSpec(
            name="answer_duckdb",
            description=(
                "Submit the final answer by running one read-only DuckDB SQL query in the "
                "task context directory. Use this for final CSV/JSON aggregations instead "
                "of execute_duckdb followed by answer. Do not use this on SQLite .db files; "
                "use answer_context_sql for SQLite-only answers or Python sqlite3 for mixed "
                "SQLite plus CSV/JSON joins."
            ),
            input_schema={"sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "extract_doc_records": ToolSpec(
            name="extract_doc_records",
            description=(
                "Extract a small structured table from a long narrative document keyed by "
                "entity ID. Use this for repeated-record markdown documents before parsing "
                "manually. Pass the fields or acronyms needed by the question or observed "
                "in the document. The extractor dynamically expands acronyms found in the "
                "document and prefers corrected/final values over initial readings."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "fields": ["field_name"],
                "max_records": 1000,
                "include_partial": False,
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document inside context. Use `start` to continue reading "
                "after a truncated preview."
            ),
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000, "start": 0},
        ),
        "search_context": ToolSpec(
            name="search_context",
            description=(
                "Search text files inside the task context for a literal query or regex. "
                "Use this for long markdown, JSON, and CSV files before reading large documents."
            ),
            input_schema={
                "query": "text or regex to search",
                "path": "optional/relative/path",
                "regex": False,
                "case_sensitive": False,
                "max_matches": 20,
                "context_chars": 120,
            },
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    }
    handlers = {
        "answer": _answer,
        "answer_context_sql": _answer_context_sql,
        "answer_duckdb": _answer_duckdb,
        "execute_context_sql": _execute_context_sql,
        "execute_duckdb": _execute_duckdb,
        "extract_doc_records": _extract_doc_records,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
        "search_context": _search_context,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
