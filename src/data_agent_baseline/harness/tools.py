from __future__ import annotations

from time import perf_counter
from typing import Any

from data_agent_baseline.benchmark.schema import (
    ANSWER_VALIDATION_FAILED,
    AnswerTable,
    AnswerValidationError,
    PublicTask,
)
from data_agent_baseline.harness.events import HarnessState
from data_agent_baseline.harness.policy import (
    ANSWER_PREPARED_ASQ_ACTION,
    EXECUTE_PREPARED_ASQ_ACTION,
    GET_SEMANTIC_CATALOG_ACTION,
    PREPARE_ASQ_ACTION,
    HarnessPolicy,
    validate_prepared_plan,
)
from data_agent_baseline.harness.session import HarnessSession
from data_agent_baseline.semantic.catalog import SemanticCatalog, load_semantic_catalog
from data_agent_baseline.semantic.compiler import compile_asq, parse_asq
from data_agent_baseline.semantic.executor import execute_semantic_plan
from data_agent_baseline.semantic.schema import CompiledSemanticPlan, SemanticModel
from data_agent_baseline.tools.registry import (
    MAX_FINAL_ROWS,
    MAX_PREVIEW_ROWS,
    ToolExecutionResult,
    ToolRegistry,
)


PREPARE_ASQ_DESCRIPTION = """- get_semantic_catalog: Return the governed dimensions, metrics, output labels, and filter operators available for the current task. Call this before prepare_asq.
  input_schema: {}
- prepare_asq: Compile an Analysis Semantic Query (ASQ) with the matched, versioned semantic model. When a deterministic compiler is available, pass only `asq`; physical paths, columns, joins, and SQL are compiler-owned.
  input_schema: {'asq': {'version': '1', 'dimensions': ['semantic.dimension'], 'metrics': ['semantic.metric'], 'filters': [{'field': 'semantic.field', 'op': '=', 'value': 'value'}], 'group_by': [], 'order_by': [], 'output': ['requested_output']}}
- execute_prepared_asq: Preview a compiled ASQ by its approved fingerprint.
  input_schema: {'fingerprint': 'returned by prepare_asq', 'limit': 200}
- answer_prepared_asq: Execute a compiled ASQ by its approved fingerprint and submit its governed output as the terminal answer.
  input_schema: {'fingerprint': 'returned by prepare_asq', 'limit': 5000}"""


class HarnessToolRegistry:
    def __init__(
        self,
        base: ToolRegistry,
        session: HarnessSession,
        policy: HarnessPolicy,
        semantic_catalog: SemanticCatalog | None = None,
    ) -> None:
        self._base = base
        self._session = session
        self._policy = policy
        self._semantic_catalog = semantic_catalog or load_semantic_catalog()
        self._compiled_plans: dict[str, CompiledSemanticPlan] = {}

    @property
    def specs(self):
        return self._base.specs

    @property
    def handlers(self):
        return self._base.handlers

    def describe_for_prompt(self) -> str:
        description = self._base.describe_for_prompt()
        if self._policy.mode == "semantic":
            return f"{PREPARE_ASQ_DESCRIPTION}\n{description}"
        return description

    def _semantic_candidates(self, task: PublicTask) -> tuple[SemanticModel, ...]:
        if self._policy.mode != "semantic":
            return ()
        return self._semantic_catalog.candidate_models(task)

    def compiler_available(self, task: PublicTask) -> bool:
        return bool(self._semantic_candidates(task))

    def register_recorded_plan(self, payload: dict[str, Any]) -> CompiledSemanticPlan:
        plan = CompiledSemanticPlan.from_dict(payload)
        self._compiled_plans[plan.fingerprint] = plan
        self._session.register_plan(plan.to_dict())
        return plan

    def _get_semantic_catalog(self, task: PublicTask) -> ToolExecutionResult:
        candidates = self._semantic_candidates(task)
        if not candidates:
            raise ValueError("No deterministic semantic model matches this task context.")
        if len(candidates) > 1:
            ids = ", ".join(model.model_id for model in candidates)
            raise ValueError(f"Multiple semantic model candidates match {task.task_id}: {ids}")
        content = candidates[0].public_catalog()
        content["match_status"] = "source_candidate"
        return ToolExecutionResult(ok=True, content=content)

    def _prepare_asq(
        self,
        task: PublicTask,
        action_input: dict[str, Any],
    ) -> ToolExecutionResult:
        candidates = self._semantic_candidates(task)
        if candidates:
            if "compiled" in action_input:
                raise ValueError(
                    "Physical SQL is compiler-owned for this task; remove `compiled` and submit only ASQ."
                )
            parsed_asq = parse_asq(action_input.get("asq"))
            required_dimensions = set(parsed_asq.dimensions) | set(parsed_asq.group_by)
            required_metrics = set(parsed_asq.metrics)
            required_filter_fields = {str(item.get("field", "")) for item in parsed_asq.filters}
            required_order_fields = {str(item.get("field", "")) for item in parsed_asq.order_by}
            model = self._semantic_catalog.match_task(
                task,
                required_dimensions=required_dimensions,
                required_metrics=required_metrics,
                required_filter_fields=required_filter_fields,
                required_order_fields=required_order_fields,
            )
            if model is None:
                decisions = [
                    decision.to_dict()
                    for decision in self._semantic_catalog.evaluate_task(
                        task,
                        required_dimensions=required_dimensions,
                        required_metrics=required_metrics,
                        required_filter_fields=required_filter_fields,
                        required_order_fields=required_order_fields,
                    )
                    if decision.source_compatible
                ]
                requirements = {
                    "dimensions": sorted(required_dimensions),
                    "metrics": sorted(required_metrics),
                    "filter_fields": sorted(required_filter_fields),
                    "order_fields": sorted(required_order_fields),
                }
                self._session.record(
                    "semantic.match.rejected",
                    {"requirements": requirements, "candidates": decisions},
                    state_after=HarnessState.VALIDATING,
                )
                reasons = [
                    reason
                    for decision in decisions
                    for reason in decision.get("reasons", [])
                    if isinstance(reason, str)
                ]
                detail = "; ".join(reasons) or "no candidate declares every required capability"
                raise ValueError(f"No capability-complete semantic model: {detail}.")
            plan = compile_asq(model, action_input.get("asq"))
            self._compiled_plans[plan.fingerprint] = plan
            plan_payload = plan.to_dict()
            self._session.register_plan(plan_payload)
            self._session.record(
                "asq.plan.compiled",
                {"plan": plan_payload},
                state_after=HarnessState.VALIDATING,
            )
            return ToolExecutionResult(
                ok=True,
                content={
                    "status": "compiled",
                    "fingerprint": plan.fingerprint,
                    "semantic_model_id": plan.semantic_model_id,
                    "semantic_model_version": plan.semantic_model_version,
                    "output_columns": list(plan.output_columns),
                    "join_path": [
                        {
                            "left": join.left,
                            "right": join.right,
                            "relationship": join.relationship,
                        }
                        for join in plan.join_path
                    ],
                    "next": "Call answer_prepared_asq with this fingerprint.",
                },
            )

        plan = validate_prepared_plan(action_input)
        plan_payload = plan.to_dict()
        self._session.register_plan(plan_payload)
        self._session.record(
            "asq.plan.prepared",
            {"plan": plan_payload},
            state_after=HarnessState.VALIDATING,
        )
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "prepared",
                "fingerprint": plan.fingerprint,
                "engine": plan.engine,
                "path": plan.path,
                "next": (
                    "Call answer_context_sql or execute_context_sql with the exact prepared query."
                    if plan.engine == "sqlite"
                    else "Call answer_duckdb or execute_duckdb with the exact prepared query."
                ),
            },
        )

    def _execute_prepared_asq(
        self,
        task: PublicTask,
        action_input: dict[str, Any],
        *,
        terminal: bool,
    ) -> ToolExecutionResult:
        fingerprint = str(action_input.get("fingerprint", ""))
        plan = self._compiled_plans.get(fingerprint)
        if plan is None:
            for payload in self._session.prepared_plans:
                if (
                    str(payload.get("fingerprint")) == fingerprint
                    and payload.get("engine") == "semantic_duckdb"
                ):
                    plan = CompiledSemanticPlan.from_dict(payload)
                    self._compiled_plans[fingerprint] = plan
                    break
        if plan is None:
            raise ValueError("Prepared semantic plan is unavailable for this fingerprint.")
        maximum = MAX_FINAL_ROWS if terminal else MAX_PREVIEW_ROWS
        default_limit = MAX_FINAL_ROWS if terminal else 200
        limit = min(int(action_input.get("limit", default_limit)), maximum)
        content = execute_semantic_plan(task, plan, limit=limit)
        if terminal and content.get("truncated") is True:
            raise ValueError("Final semantic query result was truncated; refine the ASQ.")
        if terminal and content.get("cell_truncated") is True:
            raise ValueError(
                "Final semantic query contains a truncated cell; refine the ASQ output."
            )
        if not terminal:
            return ToolExecutionResult(ok=True, content=content)
        columns = content.get("columns")
        rows = content.get("rows")
        if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
            raise ValueError("Compiled semantic query returned invalid output columns.")
        if not isinstance(rows, list):
            raise ValueError("Compiled semantic query returned invalid rows.")
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "source": ANSWER_PREPARED_ASQ_ACTION,
                "semantic_model_id": plan.semantic_model_id,
                "semantic_model_version": plan.semantic_model_version,
                "plan_fingerprint": plan.fingerprint,
                "column_count": len(columns),
                "row_count": len(rows),
            },
            is_terminal=True,
            answer=AnswerTable(columns=columns, rows=rows),
        )

    def execute(
        self,
        task: PublicTask,
        action: str,
        action_input: dict[str, Any],
    ) -> ToolExecutionResult:
        compiler_available = self.compiler_available(task)
        self._policy.authorize(
            self._session,
            action,
            action_input,
            compiler_available=compiler_available,
        )
        target_state = self._policy.state_for_action(action)
        self._session.record(
            "tool.started",
            {"action": action},
            state_after=target_state,
        )
        started_at = perf_counter()
        try:
            if action == GET_SEMANTIC_CATALOG_ACTION:
                result = self._get_semantic_catalog(task)
            elif action == PREPARE_ASQ_ACTION:
                result = self._prepare_asq(task, action_input)
            elif action == EXECUTE_PREPARED_ASQ_ACTION:
                result = self._execute_prepared_asq(task, action_input, terminal=False)
            elif action == ANSWER_PREPARED_ASQ_ACTION:
                result = self._execute_prepared_asq(task, action_input, terminal=True)
            else:
                if self._policy.mode == "raw" and action == "answer":
                    AnswerTable(
                        columns=action_input.get("columns"),  # type: ignore[arg-type]
                        rows=action_input.get("rows"),  # type: ignore[arg-type]
                    ).require_valid()
                result = self._base.execute(task, action, action_input)
            if self._policy.mode == "raw" and result.is_terminal:
                if result.answer is None:
                    raise AnswerValidationError("Terminal answer tool returned no AnswerTable.")
                result.answer.require_valid()
        except Exception as exc:
            failure_payload = {
                "action": action,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if isinstance(exc, AnswerValidationError):
                failure_payload.update(
                    {
                        "failure_type": ANSWER_VALIDATION_FAILED,
                        "answer_validation_error": str(exc),
                    }
                )
            self._session.record(
                "tool.failed",
                failure_payload,
                elapsed_ms=(perf_counter() - started_at) * 1000,
            )
            raise

        state_after = HarnessState.SUBMITTED if result.is_terminal else self._session.state
        content_summary = {
            key: result.content.get(key)
            for key in (
                "status",
                "source",
                "row_count",
                "column_count",
                "truncated",
                "semantic_model_id",
                "semantic_model_version",
                "plan_fingerprint",
            )
            if key in result.content
        }
        self._session.record(
            "tool.completed",
            {
                "action": action,
                "ok": result.ok,
                "terminal": result.is_terminal,
                "content": content_summary,
            },
            state_after=state_after,
            elapsed_ms=(perf_counter() - started_at) * 1000,
        )
        return result
