from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from data_agent_baseline.harness.events import HARNESS_VERSION, HarnessEvent, HarnessState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class HarnessVersions:
    harness: str = HARNESS_VERSION
    prompt: str = "react-v1"
    prompt_sha256: str = ""
    tools_sha256: str = ""
    semantic_model: str = "unversioned"
    policy: str = "harness-policy-v1"


@dataclass(slots=True)
class HarnessSession:
    task_id: str
    mode: str
    model: str
    max_steps: int
    attempt_number: int = 1
    prior_failure_types: tuple[str, ...] = ()
    versions: HarnessVersions = field(default_factory=HarnessVersions)
    state: HarnessState = HarnessState.CREATED
    started_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    events: list[HarnessEvent] = field(default_factory=list)
    prepared_plans: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    _started_counter: float = field(default_factory=perf_counter, repr=False)

    def record(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        state_after: HarnessState | None = None,
        elapsed_ms: float | None = None,
    ) -> HarnessEvent:
        before = self.state
        after = state_after or before
        self.state = after
        event = HarnessEvent(
            sequence=len(self.events) + 1,
            event_type=event_type,
            timestamp=_utc_now(),
            state_before=before.value,
            state_after=after.value,
            elapsed_ms=round(elapsed_ms, 3) if elapsed_ms is not None else None,
            payload=dict(payload or {}),
        )
        self.events.append(event)
        return event

    def register_plan(self, plan: dict[str, Any]) -> None:
        self.prepared_plans.append(dict(plan))

    def finalize(self, *, succeeded: bool, failure_reason: str | None = None) -> None:
        self.failure_reason = failure_reason
        final_state = HarnessState.SUBMITTED if succeeded else HarnessState.FAILED
        self.record(
            "session.completed",
            {"succeeded": succeeded, "failure_reason": failure_reason},
            state_after=final_state,
            elapsed_ms=(perf_counter() - self._started_counter) * 1000,
        )
        self.ended_at = _utc_now()

    def _totals(self) -> dict[str, int | float]:
        model_events = [event for event in self.events if event.event_type == "model.completed"]
        tool_events = [event for event in self.events if event.event_type == "tool.completed"]

        def token_total(name: str) -> int:
            return sum(int(event.payload.get(name) or 0) for event in model_events)

        return {
            "model_calls": len(model_events),
            "model_calls_with_token_usage": sum(
                1 for event in model_events if event.payload.get("total_tokens") is not None
            ),
            "tool_calls": len(tool_events),
            "prompt_tokens": token_total("prompt_tokens"),
            "completion_tokens": token_total("completion_tokens"),
            "total_tokens": token_total("total_tokens"),
            "model_elapsed_ms": round(sum(event.elapsed_ms or 0 for event in model_events), 3),
            "tool_elapsed_ms": round(sum(event.elapsed_ms or 0 for event in tool_events), 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "model": self.model,
            "max_steps": self.max_steps,
            "attempt_number": self.attempt_number,
            "prior_failure_types": list(self.prior_failure_types),
            "state": self.state.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "versions": asdict(self.versions),
            "totals": self._totals(),
            "prepared_plan_count": len(self.prepared_plans),
            "failure_reason": self.failure_reason,
            "events": [event.to_dict() for event in self.events],
        }
