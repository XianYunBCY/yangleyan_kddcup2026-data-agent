from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import (
    ANSWER_VALIDATION_FAILED,
    AnswerTable,
    AnswerValidationError,
    PublicTask,
)
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    validate_answer_sanity: bool = True


MODEL_CALL_HARD_TIMEOUT_SECONDS = 45.0
MAX_CONSECUTIVE_MODEL_FAILURES = 2
MAX_AUTO_SUBMIT_CELL_CHARS = 2000
TRANSIENT_MODEL_CONNECTION_MESSAGES = ("Connection error.", "Request timed out.")


class ModelCallTimeoutError(TimeoutError):
    pass


def is_transient_model_connection_error(error: object) -> bool:
    message = str(error)
    return any(marker in message for marker in TRANSIENT_MODEL_CONNECTION_MESSAGES)


def _is_safe_auto_submit_cell(value: object) -> bool:
    if isinstance(value, (dict, list, tuple, set)):
        return False
    if isinstance(value, str):
        if "<large nested" in value or "<truncated>" in value:
            return False
        return len(value) <= MAX_AUTO_SUBMIT_CELL_CHARS
    return len(str(value)) <= MAX_AUTO_SUBMIT_CELL_CHARS


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    unterminated_fence_match = re.search(
        r"```(?:json)?\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL
    )
    if unterminated_fence_match is not None:
        return unterminated_fence_match.group(1).strip()
    object_text = _extract_first_json_object(text)
    if object_text is not None:
        return object_text
    return text


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return None


def _escape_control_chars_in_strings(text: str) -> str:
    chars: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                chars.append(char)
                escaped = False
                continue
            if char == "\\":
                chars.append(char)
                escaped = True
                continue
            if char == '"':
                chars.append(char)
                in_string = False
                continue
            if char == "\n":
                chars.append("\\n")
                continue
            if char == "\r":
                chars.append("\\r")
                continue
            if char == "\t":
                chars.append("\\t")
                continue
            chars.append(char)
            continue

        chars.append(char)
        if char == '"':
            in_string = True
    return "".join(chars)


def _close_unbalanced_json_object(text: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
    if in_string or depth <= 0:
        return text
    return text + ("}" * depth)


def _load_single_json_object(text: str) -> dict[str, object]:
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        repaired_text = _close_unbalanced_json_object(_escape_control_chars_in_strings(text))
        payload, end = json.JSONDecoder().raw_decode(repaired_text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder and not re.fullmatch(r"}+", cleaned_remainder):
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _coerce_action_input(action: str, action_input: object) -> dict[str, object]:
    if isinstance(action_input, dict):
        return action_input
    if isinstance(action_input, str):
        if action == "execute_python":
            return {"code": action_input}
        if action in {"execute_duckdb", "answer_duckdb"}:
            return {"sql": action_input}
        if action in {"read_csv", "read_json", "read_doc", "inspect_sqlite_schema"}:
            return {"path": action_input}
        if action == "search_context":
            return {"query": action_input}
        if action == "answer":
            try:
                parsed = json.loads(action_input)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("action_input must be a JSON object.")


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    action_input = _coerce_action_input(action, action_input)

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _latest_table_observation(self, state: AgentRuntimeState) -> dict[str, object] | None:
        for step in reversed(state.steps):
            content = step.observation.get("content")
            if not isinstance(content, dict):
                continue
            columns = content.get("columns")
            rows = content.get("rows")
            if isinstance(columns, list) and isinstance(rows, list):
                return content
        return None

    def _repair_answer_input(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
    ) -> dict[str, object]:
        return action_input

    def _answer_validation_error(
        self,
        task: PublicTask,
        original_input: dict[str, object],
        repaired_input: dict[str, object],
        raw_response: str,
    ) -> str | None:
        del task
        del original_input
        del raw_response
        if not self.config.validate_answer_sanity:
            return None
        return AnswerTable(
            columns=repaired_input.get("columns"),  # type: ignore[arg-type]
            rows=repaired_input.get("rows"),  # type: ignore[arg-type]
        ).validation_error()

    @staticmethod
    def _answer_validation_observation(error: str) -> dict[str, object]:
        return {
            "ok": False,
            "failure_type": ANSWER_VALIDATION_FAILED,
            "answer_validation_error": error,
            "error": error,
        }

    @staticmethod
    def _remember_answer_validation_error(state: AgentRuntimeState, error: str) -> None:
        state.answer_validation_error = error

    def _try_auto_submit_latest_table(self, task: PublicTask, state: AgentRuntimeState) -> bool:
        latest_table = self._latest_table_observation(state)
        if latest_table is None:
            return False
        columns = latest_table.get("columns")
        rows = latest_table.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            return False
        if latest_table.get("cell_truncated") is True:
            return False
        if not columns:
            return False
        if not rows:
            return False
        if any(
            not isinstance(row, list) or any(not _is_safe_auto_submit_cell(cell) for cell in row)
            for row in rows
        ):
            return False
        action_input: dict[str, object] = {
            "columns": [str(column) for column in columns],
            "rows": [list(row) for row in rows if isinstance(row, list)],
        }
        if len(action_input["rows"]) != len(rows):
            return False
        action_input = self._repair_answer_input(task, state, action_input)
        validation_error = self._answer_validation_error(
            task,
            action_input,
            action_input,
            "__auto_submit_latest_table__",
        )
        if validation_error is not None:
            self._remember_answer_validation_error(state, validation_error)
            state.steps.append(
                StepRecord(
                    step_index=len(state.steps) + 1,
                    thought="Skipping auto-submit because the latest table is not a valid final answer.",
                    action="answer",
                    action_input=action_input,
                    raw_response="__auto_submit_latest_table__",
                    observation=self._answer_validation_observation(validation_error),
                    ok=False,
                )
            )
            return False
        try:
            tool_result = self.tools.execute(task, "answer", action_input)
        except Exception as exc:
            if isinstance(exc, AnswerValidationError):
                self._remember_answer_validation_error(state, str(exc))
            observation = (
                self._answer_validation_observation(str(exc))
                if isinstance(exc, AnswerValidationError)
                else {"ok": False, "error": str(exc)}
            )
            ok = False
        else:
            observation = {
                "ok": tool_result.ok,
                "tool": "answer",
                "content": tool_result.content,
            }
            ok = tool_result.ok
            if tool_result.is_terminal:
                state.answer = tool_result.answer
                state.answer_validation_error = None

        state.steps.append(
            StepRecord(
                step_index=len(state.steps) + 1,
                thought="Auto-submitting the latest table result as a best-effort answer.",
                action="answer",
                action_input=action_input,
                raw_response="__auto_submit_latest_table__",
                observation=observation,
                ok=ok,
            )
        )
        return state.answer is not None

    def _try_submit_action_input(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        action_input: dict[str, object],
        *,
        thought: str,
        raw_response: str,
    ) -> bool:
        action_input = self._repair_answer_input(task, state, action_input)
        validation_error = self._answer_validation_error(
            task, action_input, action_input, raw_response
        )
        step_index = len(state.steps) + 1
        if validation_error is not None:
            self._remember_answer_validation_error(state, validation_error)
            state.steps.append(
                StepRecord(
                    step_index=step_index,
                    thought=thought,
                    action="answer",
                    action_input=action_input,
                    raw_response=raw_response,
                    observation=self._answer_validation_observation(validation_error),
                    ok=False,
                )
            )
            return False
        try:
            tool_result = self.tools.execute(task, "answer", action_input)
        except Exception as exc:
            if isinstance(exc, AnswerValidationError):
                self._remember_answer_validation_error(state, str(exc))
            state.steps.append(
                StepRecord(
                    step_index=step_index,
                    thought=thought,
                    action="answer",
                    action_input=action_input,
                    raw_response=raw_response,
                    observation=(
                        self._answer_validation_observation(str(exc))
                        if isinstance(exc, AnswerValidationError)
                        else {"ok": False, "error": str(exc)}
                    ),
                    ok=False,
                )
            )
            return False

        state.steps.append(
            StepRecord(
                step_index=step_index,
                thought=thought,
                action="answer",
                action_input=action_input,
                raw_response=raw_response,
                observation={
                    "ok": tool_result.ok,
                    "tool": "answer",
                    "content": tool_result.content,
                },
                ok=tool_result.ok,
            )
        )
        if tool_result.is_terminal:
            state.answer = tool_result.answer
            state.answer_validation_error = None
        return state.answer is not None

    def _try_force_submit_from_observations(
        self, task: PublicTask, state: AgentRuntimeState
    ) -> bool:
        return False

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        next_step_index = len(state.steps) + 1
        remaining_steps = self.config.max_steps - len(state.steps)
        if 0 < remaining_steps <= 3:
            if remaining_steps <= 1:
                message = (
                    f"You are about to take the final allowed ReAct step "
                    f"({next_step_index} of {self.config.max_steps}). You must call a "
                    "terminal answer tool now (`answer`, `answer_duckdb`, or "
                    "`answer_context_sql`, or `answer_prepared_asq`) using the best final table supported by the "
                    "observations. Do not call any exploratory or non-terminal tool."
                )
            else:
                message = (
                    f"You are about to take step {next_step_index} of "
                    f"{self.config.max_steps}. You have {remaining_steps} step(s) left. "
                    "If you have enough information or a best-effort result, call a "
                    "terminal answer tool now. Do not run another exploratory tool unless "
                    "it is strictly necessary to form the final table."
                )
            messages.append(
                ModelMessage(
                    role="user",
                    content=message,
                )
            )
        return messages

    def _build_final_answer_messages(
        self, task: PublicTask, state: AgentRuntimeState
    ) -> list[ModelMessage]:
        messages = self._build_messages(task, state)
        messages.append(
            ModelMessage(
                role="user",
                content=(
                    "The ReAct tool budget is exhausted. You have now seen all available "
                    "observations, including the result of the last tool call. Your next "
                    "message must be a JSON object that calls `answer`, `answer_duckdb`, "
                    "`answer_context_sql`, or `answer_prepared_asq`. Do not call list_context, read_doc, "
                    "execute_python, or any non-terminal exploratory tool. If the "
                    "observations contain a computed final value, submit it as the final "
                    "answer table."
                ),
            )
        )
        return messages

    def _complete_model(self, messages: list[ModelMessage]) -> str:
        result_queue: queue.Queue[tuple[str, str | Exception]] = queue.Queue(maxsize=1)

        def _complete_in_daemon_thread() -> None:
            try:
                result_queue.put(("ok", self.model.complete(messages)))
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("error", exc))

        thread = threading.Thread(target=_complete_in_daemon_thread, daemon=True)
        thread.start()
        thread.join(timeout=MODEL_CALL_HARD_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise ModelCallTimeoutError(
                f"Model request exceeded hard timeout of {MODEL_CALL_HARD_TIMEOUT_SECONDS:g} seconds."
            )
        status, payload = result_queue.get_nowait()
        if status == "ok":
            return str(payload)
        if isinstance(payload, Exception):
            raise payload
        raise RuntimeError(str(payload))

    def _error_action_name(self, exc: Exception) -> str:
        if isinstance(exc, AnswerValidationError):
            return "__answer_validation_failed__"
        if isinstance(exc, ModelCallTimeoutError):
            return "__model_timeout__"
        message = str(exc)
        if message.startswith("Model request failed:") or "Model response missing" in message:
            return "__model_error__"
        if isinstance(exc, ValueError):
            return "__parse_error__"
        return "__tool_error__"

    def _repair_terminal_sql_answer(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        answer: AnswerTable,
        raw_response: str,
    ) -> tuple[AnswerTable, str | None]:
        original_input: dict[str, object] = {
            "columns": list(answer.columns),
            "rows": [list(row) for row in answer.rows],
        }
        repaired_input = self._repair_answer_input(task, state, original_input)
        validation_error = self._answer_validation_error(
            task,
            original_input,
            repaired_input,
            raw_response,
        )
        if validation_error is not None:
            return answer, validation_error
        repaired_columns = repaired_input.get("columns")
        repaired_rows = repaired_input.get("rows")
        if not isinstance(repaired_columns, list) or not all(
            isinstance(item, str) for item in repaired_columns
        ):
            return answer, None
        if not isinstance(repaired_rows, list):
            return answer, None
        return AnswerTable(
            columns=list(repaired_columns), rows=[list(row) for row in repaired_rows]
        ), None

    def _try_final_sql_submission(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        step_index: int,
        thought: str,
        action: str,
        action_input: dict[str, Any],
        raw_response: str,
    ) -> bool:
        final_action_by_model_action = {
            "answer_prepared_asq": "answer_prepared_asq",
            "answer_context_sql": "answer_context_sql",
            "answer_duckdb": "answer_duckdb",
            "execute_context_sql": "answer_context_sql",
            "execute_duckdb": "answer_duckdb",
        }
        final_action = final_action_by_model_action.get(action)
        if final_action is None:
            return False
        sql_text = str(action_input.get("sql", ""))
        if action in {"execute_context_sql", "execute_duckdb"} and re.search(
            r"\b(?:sqlite_master|information_schema|pragma|show\s+tables|describe)\b",
            sql_text,
            flags=re.IGNORECASE,
        ):
            return False
        try:
            tool_result = self.tools.execute(task, final_action, action_input)
            terminal_answer = tool_result.answer
            validation_error = None
            if tool_result.is_terminal and terminal_answer is not None:
                terminal_answer, validation_error = self._repair_terminal_sql_answer(
                    task,
                    state,
                    terminal_answer,
                    raw_response,
                )
            observation = {
                "ok": tool_result.ok and validation_error is None,
                "tool": final_action,
                "content": tool_result.content
                if validation_error is None
                else {"error": validation_error, "source": tool_result.content},
            }
            if validation_error is not None:
                self._remember_answer_validation_error(state, validation_error)
                observation.update(
                    {
                        "failure_type": ANSWER_VALIDATION_FAILED,
                        "answer_validation_error": validation_error,
                    }
                )
            state.steps.append(
                StepRecord(
                    step_index=step_index,
                    thought=thought,
                    action=final_action,
                    action_input=action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok and validation_error is None,
                )
            )
            if tool_result.is_terminal and validation_error is None:
                state.answer = terminal_answer
                state.answer_validation_error = None
            return True
        except Exception as exc:
            if isinstance(exc, AnswerValidationError):
                self._remember_answer_validation_error(state, str(exc))
            state.steps.append(
                StepRecord(
                    step_index=step_index,
                    thought=thought,
                    action=final_action,
                    action_input=action_input,
                    raw_response=raw_response,
                    observation=(
                        self._answer_validation_observation(str(exc))
                        if isinstance(exc, AnswerValidationError)
                        else {"ok": False, "error": str(exc)}
                    ),
                    ok=False,
                )
            )
            return True

    def _try_final_answer_step(self, task: PublicTask, state: AgentRuntimeState) -> bool:
        if state.answer is not None:
            return False

        attempted = False
        for _ in range(3):
            if state.answer is not None:
                break
            attempted = True
            raw_response = ""
            step_index = len(state.steps) + 1
            try:
                raw_response = self._complete_model(self._build_final_answer_messages(task, state))
                model_step = parse_model_step(raw_response)
                if model_step.action != "answer":
                    if model_step.action == "execute_python":
                        try:
                            tool_result = self.tools.execute(
                                task, "execute_python", model_step.action_input
                            )
                            observation = {
                                "ok": tool_result.ok,
                                "tool": "execute_python",
                                "content": tool_result.content,
                            }
                            state.steps.append(
                                StepRecord(
                                    step_index=step_index,
                                    thought=model_step.thought,
                                    action="execute_python",
                                    action_input=model_step.action_input,
                                    raw_response=raw_response,
                                    observation=observation,
                                    ok=tool_result.ok,
                                )
                            )
                            if tool_result.ok:
                                self._try_force_submit_from_observations(task, state)
                            continue
                        except Exception as exc:
                            state.steps.append(
                                StepRecord(
                                    step_index=step_index,
                                    thought=model_step.thought,
                                    action="execute_python",
                                    action_input=model_step.action_input,
                                    raw_response=raw_response,
                                    observation={"ok": False, "error": str(exc)},
                                    ok=False,
                                )
                            )
                            continue
                    if self._try_final_sql_submission(
                        task,
                        state,
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                    ):
                        continue
                    observation = {
                        "ok": False,
                        "error": (
                            "Finalization step may only call the answer tool. "
                            "No further data-gathering or computation tools are available; "
                            "submit the best answer supported by the existing observations."
                        ),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                    )
                    continue

                action_input = self._repair_answer_input(task, state, model_step.action_input)
                validation_error = self._answer_validation_error(
                    task,
                    model_step.action_input,
                    action_input,
                    raw_response,
                )
                if validation_error is not None:
                    self._remember_answer_validation_error(state, validation_error)
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action="answer",
                            action_input=action_input,
                            raw_response=raw_response,
                            observation=self._answer_validation_observation(validation_error),
                            ok=False,
                        )
                    )
                    continue
                tool_result = self.tools.execute(task, "answer", action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": "answer",
                    "content": tool_result.content,
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action="answer",
                        action_input=action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                )
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    state.answer_validation_error = None
            except Exception as exc:
                if isinstance(exc, AnswerValidationError):
                    self._remember_answer_validation_error(state, str(exc))
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action=self._error_action_name(exc),
                        action_input={},
                        raw_response=raw_response,
                        observation=(
                            self._answer_validation_observation(str(exc))
                            if isinstance(exc, AnswerValidationError)
                            else {"ok": False, "error": str(exc)}
                        ),
                        ok=False,
                    )
                )
        return attempted

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        consecutive_model_failures = 0
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = ""
            try:
                raw_response = self._complete_model(self._build_messages(task, state))
                consecutive_model_failures = 0
                model_step = parse_model_step(raw_response)
                action_input = model_step.action_input
                if model_step.action == "answer":
                    original_action_input = action_input
                    action_input = self._repair_answer_input(task, state, action_input)
                    validation_error = self._answer_validation_error(
                        task,
                        original_action_input,
                        action_input,
                        raw_response,
                    )
                    if validation_error is not None:
                        self._remember_answer_validation_error(state, validation_error)
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action="answer",
                                action_input=action_input,
                                raw_response=raw_response,
                                observation=self._answer_validation_observation(validation_error),
                                ok=False,
                            )
                        )
                        continue
                tool_result = self.tools.execute(task, model_step.action, action_input)
                terminal_answer = tool_result.answer
                validation_error = None
                if (
                    model_step.action in {"answer_context_sql", "answer_duckdb"}
                    and tool_result.is_terminal
                    and terminal_answer is not None
                ):
                    terminal_answer, validation_error = self._repair_terminal_sql_answer(
                        task,
                        state,
                        terminal_answer,
                        raw_response,
                    )
                observation = {
                    "ok": tool_result.ok and validation_error is None,
                    "tool": model_step.action,
                    "content": tool_result.content
                    if validation_error is None
                    else {"error": validation_error, "source": tool_result.content},
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok and validation_error is None,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal and validation_error is None:
                    state.answer = terminal_answer
                    state.answer_validation_error = None
                    break
            except Exception as exc:
                if isinstance(exc, AnswerValidationError):
                    self._remember_answer_validation_error(state, str(exc))
                error_action = self._error_action_name(exc)
                if (
                    isinstance(exc, ModelCallTimeoutError)
                    or str(exc).startswith("Model request failed:")
                    or "Model response missing" in str(exc)
                ):
                    consecutive_model_failures += 1
                else:
                    consecutive_model_failures = 0
                observation = (
                    self._answer_validation_observation(str(exc))
                    if isinstance(exc, AnswerValidationError)
                    else {"ok": False, "error": str(exc)}
                )
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action=error_action,
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                if error_action == "__model_error__" and is_transient_model_connection_error(
                    exc
                ):
                    state.failure_reason = f"{error_action}: {exc}"
                    break
                if consecutive_model_failures >= MAX_CONSECUTIVE_MODEL_FAILURES:
                    state.failure_reason = f"{error_action}: {exc}"
                    break

        if state.answer is None and state.failure_reason is None and state.steps:
            self._try_final_answer_step(task, state)

        if state.answer is None and state.failure_reason is None:
            self._try_force_submit_from_observations(task, state)

        if state.answer is None and state.failure_reason is None:
            self._try_auto_submit_latest_table(task, state)

        if (
            state.answer is None
            and state.failure_reason is not None
            and state.failure_reason.startswith(("__model_error__:", "__model_timeout__:"))
            and (
                self._try_force_submit_from_observations(task, state)
                or self._try_auto_submit_latest_table(task, state)
            )
        ):
            state.failure_reason = None

        if state.answer is None and state.failure_reason is None:
            if state.answer_validation_error is not None:
                state.failure_reason = (
                    f"{ANSWER_VALIDATION_FAILED}: {state.answer_validation_error}"
                )
            else:
                state.failure_reason = "Agent did not submit an answer within max_steps."

        final_answer_validation_error = None
        if self.config.validate_answer_sanity and state.answer is not None:
            final_answer_validation_error = state.answer.validation_error()
            if final_answer_validation_error is not None and state.failure_reason is None:
                state.failure_reason = (
                    f"{ANSWER_VALIDATION_FAILED}: {final_answer_validation_error}"
                )
        elif state.answer is None:
            final_answer_validation_error = state.answer_validation_error

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            answer_validation_error=final_answer_validation_error,
            validate_answer_sanity=self.config.validate_answer_sanity,
        )
