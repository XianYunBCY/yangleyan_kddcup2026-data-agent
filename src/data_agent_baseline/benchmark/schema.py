from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANSWER_VALIDATION_FAILED = "answer_validation_failed"


class AnswerValidationError(ValueError):
    """Raised when a submitted answer is not a structurally sane result table."""

    failure_type = ANSWER_VALIDATION_FAILED


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    difficulty: str
    question: str


@dataclass(frozen=True, slots=True)
class TaskAssets:
    task_dir: Path
    context_dir: Path


@dataclass(frozen=True, slots=True)
class PublicTask:
    record: TaskRecord
    assets: TaskAssets

    @property
    def task_id(self) -> str:
        return self.record.task_id

    @property
    def difficulty(self) -> str:
        return self.record.difficulty

    @property
    def question(self) -> str:
        return self.record.question

    @property
    def task_dir(self) -> Path:
        return self.assets.task_dir

    @property
    def context_dir(self) -> Path:
        return self.assets.context_dir


@dataclass(frozen=True, slots=True)
class AnswerTable:
    columns: list[str]
    rows: list[list[Any]]

    def validation_error(self) -> str | None:
        if not isinstance(self.columns, list) or not self.columns:
            return "Answer table must contain at least one column."
        if not all(isinstance(column, str) for column in self.columns):
            return "Answer table columns must be strings."
        if not isinstance(self.rows, list) or not self.rows:
            return "Answer table must contain at least one row."
        for row in self.rows:
            if not isinstance(row, list):
                return "Each answer row must be a list."
            if len(row) != len(self.columns):
                return "Each answer row must match the number of columns."
        if all(cell is None for row in self.rows for cell in row):
            return "Answer table must contain at least one non-NULL value."
        return None

    def require_valid(self) -> None:
        error = self.validation_error()
        if error is not None:
            raise AnswerValidationError(error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }
