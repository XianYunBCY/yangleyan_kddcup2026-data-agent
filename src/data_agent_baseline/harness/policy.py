from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from data_agent_baseline.harness.events import HarnessState
from data_agent_baseline.harness.session import HarnessSession


POLICY_VERSION = "harness-policy-v1"
PREPARE_ASQ_ACTION = "prepare_asq"
GET_SEMANTIC_CATALOG_ACTION = "get_semantic_catalog"
EXECUTE_PREPARED_ASQ_ACTION = "execute_prepared_asq"
ANSWER_PREPARED_ASQ_ACTION = "answer_prepared_asq"

DISCOVERY_ACTIONS = {
    "list_context",
    "read_csv",
    "read_json",
    "read_doc",
    "search_context",
    "extract_doc_records",
    "inspect_sqlite_schema",
    GET_SEMANTIC_CATALOG_ACTION,
}
SQL_ACTIONS = {
    "execute_context_sql",
    "answer_context_sql",
    "execute_duckdb",
    "answer_duckdb",
}


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()


def query_fingerprint(*, engine: str, path: str | None, sql: str) -> str:
    payload = {
        "engine": engine,
        "path": path or None,
        "sql": _normalize_sql(sql),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedQueryPlan:
    asq: dict[str, Any]
    engine: str
    path: str | None
    sql: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_prepared_plan(action_input: dict[str, Any]) -> PreparedQueryPlan:
    asq = action_input.get("asq")
    compiled = action_input.get("compiled")
    if not isinstance(asq, dict):
        raise ValueError("prepare_asq.asq must be an object.")
    if str(asq.get("version", "")) != "1":
        raise ValueError("prepare_asq.asq.version must be '1'.")
    output = asq.get("output")
    if (
        not isinstance(output, list)
        or not output
        or not all(isinstance(item, str) for item in output)
    ):
        raise ValueError(
            "prepare_asq.asq.output must be a non-empty list of semantic output names."
        )
    for list_field in ("dimensions", "metrics", "group_by"):
        value = asq.get(list_field, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(
                f"prepare_asq.asq.{list_field} must be a list of semantic names when provided."
            )
    for list_field in ("filters", "order_by"):
        value = asq.get(list_field, [])
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(
                f"prepare_asq.asq.{list_field} must be a list of objects when provided."
            )
    limit = asq.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("prepare_asq.asq.limit must be a positive integer when provided.")

    if not isinstance(compiled, dict):
        raise ValueError("prepare_asq.compiled must be an object.")
    engine = str(compiled.get("engine", "")).lower()
    if engine not in {"sqlite", "duckdb"}:
        raise ValueError("prepare_asq.compiled.engine must be 'sqlite' or 'duckdb'.")
    path_value = compiled.get("path")
    path = str(path_value) if path_value is not None else None
    if engine == "sqlite" and not path:
        raise ValueError("prepare_asq.compiled.path is required for SQLite queries.")
    sql = str(compiled.get("sql", "")).strip()
    if not sql.lower().startswith(("select", "with")):
        raise ValueError("prepare_asq.compiled.sql must be a read-only SELECT or WITH query.")
    return PreparedQueryPlan(
        asq=dict(asq),
        engine=engine,
        path=path,
        sql=sql,
        fingerprint=query_fingerprint(engine=engine, path=path, sql=sql),
    )


def action_query_fingerprint(action: str, action_input: dict[str, Any]) -> str | None:
    if action not in SQL_ACTIONS:
        return None
    engine = "sqlite" if action in {"execute_context_sql", "answer_context_sql"} else "duckdb"
    path = str(action_input.get("path")) if engine == "sqlite" else None
    return query_fingerprint(engine=engine, path=path, sql=str(action_input.get("sql", "")))


@dataclass(frozen=True, slots=True)
class HarnessPolicy:
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {"raw", "semantic"}:
            raise ValueError("Harness mode must be 'raw' or 'semantic'.")

    def authorize(
        self,
        session: HarnessSession,
        action: str,
        action_input: dict[str, Any],
        *,
        compiler_available: bool = False,
    ) -> None:
        if session.state in {HarnessState.SUBMITTED, HarnessState.FAILED}:
            raise PermissionError(
                f"Session is terminal ({session.state.value}); no more tools are allowed."
            )
        if self.mode == "raw":
            return
        if compiler_available:
            if action in SQL_ACTIONS or action in {"execute_python", "answer"}:
                raise PermissionError(
                    "A deterministic semantic compiler is available for this task. Use "
                    "get_semantic_catalog, prepare_asq, and answer_prepared_asq; raw SQL, "
                    "arbitrary Python, and manual answer submission are disabled."
                )
            if action in {EXECUTE_PREPARED_ASQ_ACTION, ANSWER_PREPARED_ASQ_ACTION}:
                fingerprint = str(action_input.get("fingerprint", ""))
                approved = {str(plan.get("fingerprint")) for plan in session.prepared_plans}
                if fingerprint not in approved:
                    raise PermissionError("Unknown or unapproved prepared ASQ fingerprint.")
            return
        if action == "execute_python":
            raise PermissionError("Semantic mode disables arbitrary Python execution.")
        if action in SQL_ACTIONS:
            fingerprint = action_query_fingerprint(action, action_input)
            approved = {str(plan.get("fingerprint")) for plan in session.prepared_plans}
            if fingerprint not in approved:
                raise PermissionError(
                    "Semantic mode requires this exact SQL and source to be registered with "
                    "prepare_asq before execution."
                )

    def state_for_action(self, action: str) -> HarnessState:
        if action in DISCOVERY_ACTIONS:
            return HarnessState.DISCOVERING
        if action == PREPARE_ASQ_ACTION:
            return HarnessState.COMPILING
        if action in {EXECUTE_PREPARED_ASQ_ACTION, ANSWER_PREPARED_ASQ_ACTION}:
            return HarnessState.EXECUTING
        if action in SQL_ACTIONS or action == "execute_python":
            return HarnessState.EXECUTING
        if action == "answer":
            return HarnessState.VALIDATING
        return HarnessState.PLANNING
