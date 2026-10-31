from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


HARNESS_VERSION = "1.0"


class HarnessState(str, Enum):
    CREATED = "created"
    DISCOVERING = "discovering"
    PLANNING = "planning"
    COMPILING = "compiling"
    VALIDATING = "validating"
    EXECUTING = "executing"
    SUBMITTED = "submitted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    sequence: int
    event_type: str
    timestamp: str
    state_before: str
    state_after: str
    elapsed_ms: float | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
