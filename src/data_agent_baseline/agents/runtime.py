from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import ANSWER_VALIDATION_FAILED, AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None
    answer_validation_error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None
    answer_validation_error: str | None = None
    validate_answer_sanity: bool = False

    @property
    def effective_answer_validation_error(self) -> str | None:
        if self.answer_validation_error is not None:
            return self.answer_validation_error
        if self.validate_answer_sanity and self.answer is not None:
            return self.answer.validation_error()
        return None

    @property
    def effective_failure_reason(self) -> str | None:
        if self.failure_reason is not None:
            return self.failure_reason
        validation_error = self.effective_answer_validation_error
        if validation_error is not None:
            return f"{ANSWER_VALIDATION_FAILED}: {validation_error}"
        return None

    @property
    def succeeded(self) -> bool:
        return (
            self.answer is not None
            and self.effective_failure_reason is None
            and self.effective_answer_validation_error is None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.effective_failure_reason,
            "failure_type": (
                ANSWER_VALIDATION_FAILED
                if self.effective_answer_validation_error is not None
                else None
            ),
            "answer_validation_error": self.effective_answer_validation_error,
            "succeeded": self.succeeded,
        }
