"""LLM client abstraction with OpenAI Chat Completions + Structured Outputs."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["customer", "business"]},
                    "text": {"type": "string"},
                },
                "required": ["role", "text"],
                "additionalProperties": False,
            },
        },
        "actual_outcome": {"type": "string"},
        "trajectory_notes": {"type": "string"},
    },
    "required": ["messages", "actual_outcome", "trajectory_notes"],
    "additionalProperties": False,
}


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    model_name: str = ""


@dataclass
class LLMResult:
    content: str
    usage: UsageStats = field(default_factory=UsageStats)
    refusal: str | None = None
    finish_reason: str = ""
    elapsed_seconds: float = 0.0


class LLMClient(ABC):
    """Abstract base — add Anthropic, Gemini, etc. by subclassing."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        """Return the raw text response from the LLM."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the client is configured and ready."""
        ...


class OpenAIClient(LLMClient):
    """OpenAI implementation using Chat Completions with Structured Outputs."""

    _STRUCTURED_PREFIX = ("gpt-4o", "gpt-4", "o1", "o3", "gpt-5")

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    def _supports_structured_outputs(self) -> bool:
        base = self._model.split(":")[0]
        return any(base.startswith(p) for p in self._STRUCTURED_PREFIX)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        client = self._get_client()
        last_error: Exception | None = None
        use_structured = self._supports_structured_outputs()

        for attempt in range(self._max_retries):
            t0 = time.monotonic()
            try:
                result = self._call_chat_completions(client, system_prompt, user_prompt, use_structured)
                result.elapsed_seconds = time.monotonic() - t0
                logger.info(
                    "LLM response | model=%s | finish=%s | in=%d out=%d total=%d | elapsed=%.2fs",
                    self._model,
                    result.finish_reason,
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.usage.total_tokens,
                    result.elapsed_seconds,
                )
                return result

            except Exception as exc:
                cls_name = type(exc).__name__
                if any(t in cls_name for t in ("AuthenticationError", "PermissionDenied")):
                    raise
                if "invalid_api_key" in str(exc).lower():
                    raise

                last_error = exc
                elapsed = time.monotonic() - t0
                jitter = 0.1 * attempt
                wait = (2**attempt) + jitter
                logger.warning(
                    "LLM request failed (attempt %d/%d) after %.2fs: %s — retrying in %.1fs",
                    attempt + 1,
                    self._max_retries,
                    elapsed,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM request failed after {self._max_retries} attempts: {last_error}"
        )

    def _call_chat_completions(
        self,
        client: Any,
        system_prompt: str,
        user_prompt: str,
        use_structured: bool,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": 16000,
        }
        if use_structured:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_conversation_response",
                    "strict": True,
                    "schema": _LLM_RESPONSE_SCHEMA,
                },
            }

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        refusal = getattr(msg, "refusal", None)
        finish_reason = choice.finish_reason or ""

        usage = UsageStats(model_name=self._model)
        if response.usage:
            usage.input_tokens = response.usage.prompt_tokens
            usage.output_tokens = response.usage.completion_tokens
            usage.total_tokens = response.usage.total_tokens
            cached = getattr(response.usage, "prompt_tokens_details", None)
            if cached:
                usage.cached_input_tokens = getattr(cached, "cached_tokens", 0)
        usage.request_count = 1

        return LLMResult(content=content, usage=usage, refusal=refusal, finish_reason=finish_reason)


class MockLLMClient(LLMClient):
    """Returns pre-built fake responses for tests / dry-run without API key."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    def is_available(self) -> bool:
        return True

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        import re as _re

        idx = self._call_count % max(len(self._responses), 1)
        self._call_count += 1

        if self._responses:
            content = self._responses[idx]
        else:
            import uuid as _uuid
            # Try to match the requested message count from the prompt
            match = _re.search(r"בדיוק\s+(\d+)\s+הודעות", user_prompt)
            n = int(match.group(1)) if match else 7
            unique_id = _uuid.uuid4().hex[:8]
            domain_match = _re.search(r"תחום:\s*(\S+)", user_prompt)
            domain_hint = domain_match.group(1) if domain_match else "שירות"
            msgs = []
            for i in range(n):
                role = "customer" if i % 2 == 0 else "business"
                suffix = f"[{unique_id}-{i}]"
                if role == "customer":
                    text = f"שלום, אני מחפש {domain_hint} ואני צריך עזרה {suffix}"
                else:
                    text = f"ברוך הבא, אשמח לסייע בנושא {domain_hint} {suffix}"
                msgs.append({"role": role, "text": text})
            content = json.dumps(
                {
                    "messages": msgs,
                    "actual_outcome": "pending",
                    "trajectory_notes": "שיחת בדיקה מדומה",
                },
                ensure_ascii=False,
            )

        usage = UsageStats(
            input_tokens=500,
            output_tokens=300,
            total_tokens=800,
            request_count=1,
            model_name="mock",
        )
        return LLMResult(content=content, usage=usage)


def create_client(config: Any) -> LLMClient:
    """Factory — returns the right LLM client based on configuration."""
    provider = getattr(config, "llm_provider", "openai").lower()

    if provider == "openai":
        return OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            timeout=config.request_timeout_seconds,
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Supported providers: openai"
    )
