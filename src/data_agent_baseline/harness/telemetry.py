from __future__ import annotations

from time import perf_counter

from data_agent_baseline.agents.model import ModelAdapter, ModelCompletion, ModelMessage
from data_agent_baseline.harness.session import HarnessSession


class InstrumentedModelAdapter:
    def __init__(self, model: ModelAdapter, session: HarnessSession) -> None:
        self._model = model
        self._session = session

    def complete(self, messages: list[ModelMessage]) -> str:
        request_chars = sum(len(message.content) for message in messages)
        self._session.record(
            "model.started",
            {"message_count": len(messages), "request_chars": request_chars},
        )
        started_at = perf_counter()
        try:
            complete_with_metadata = getattr(self._model, "complete_with_metadata", None)
            if callable(complete_with_metadata):
                completion = complete_with_metadata(messages)
            else:
                completion = ModelCompletion(content=self._model.complete(messages))
        except Exception as exc:
            elapsed_ms = (perf_counter() - started_at) * 1000
            self._session.record(
                "model.failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
                elapsed_ms=elapsed_ms,
            )
            raise

        elapsed_ms = (perf_counter() - started_at) * 1000
        self._session.record(
            "model.completed",
            {
                "response_chars": len(completion.content),
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.total_tokens,
                "response_model": completion.model,
                "response_id": completion.response_id,
            },
            elapsed_ms=elapsed_ms,
        )
        return completion.content
