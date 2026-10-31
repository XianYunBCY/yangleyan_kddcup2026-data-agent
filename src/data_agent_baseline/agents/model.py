from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI

MODEL_REQUEST_TIMEOUT_SECONDS = 30.0
MODEL_MAX_RETRIES = 1


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


@dataclass(frozen=True, slots=True)
class ModelCompletion:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None
    response_id: str | None = None


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def complete_with_metadata(self, messages: list[ModelMessage]) -> ModelCompletion:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
            max_retries=MODEL_MAX_RETRIES,
        )

        last_error: Exception | None = None
        for attempt in range(MODEL_MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": message.role, "content": message.content} for message in messages
                    ],
                    temperature=self.temperature,
                )
                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Model response missing choices.")
                content = choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Model response missing text content.")
                usage = response.usage
                return ModelCompletion(
                    content=content,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                    model=getattr(response, "model", None) or self.model,
                    response_id=getattr(response, "id", None),
                )
            except (APIError, RuntimeError) as exc:
                last_error = exc
                if attempt >= MODEL_MAX_RETRIES:
                    break
                time.sleep(0.5 * (attempt + 1))

        if isinstance(last_error, APIError):
            raise RuntimeError(f"Model request failed: {last_error}") from last_error
        raise RuntimeError(str(last_error) if last_error is not None else "Model request failed.")

    def complete(self, messages: list[ModelMessage]) -> str:
        return self.complete_with_metadata(messages).content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
