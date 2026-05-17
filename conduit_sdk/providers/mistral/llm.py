"""
MistralLLMClient — adapter for Mistral AI Chat API.

Uses the official ``mistralai`` Python SDK.

Supported models: mistral-large-latest, mistral-small-latest,
                  open-mistral-nemo, codestral-latest, …

Install: pip install llm-conduit[mistral]
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from conduit_sdk.clients.llm import LLMClient
from conduit_sdk.core.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason, LLMResponse

if TYPE_CHECKING:
    pass


def _require_mistral():
    try:
        import mistralai  # noqa: PLC0415

        return mistralai
    except ImportError:
        raise ImportError(
            "Mistral provider requires the mistralai package. "
            "Install it with: pip install llm-conduit[mistral]"
        ) from None


def _wrap_mistral_error(exc: Exception, provider: str = "mistral") -> ProviderError:
    try:
        exc_type = type(exc).__name__
        if "Unauthorized" in exc_type or "Authentication" in exc_type:
            return AuthenticationError(str(exc), provider=provider)
        if "RateLimit" in exc_type or "TooManyRequests" in exc_type:
            return RateLimitError(str(exc), provider=provider)
        if "Timeout" in exc_type or "Deadline" in exc_type:
            return TimeoutError(str(exc), provider=provider)
        if hasattr(exc, "status_code"):
            return ProviderError(str(exc), provider=provider, status_code=exc.status_code)
    except ImportError:
        pass
    return ProviderError(str(exc), provider=provider)


def _map_finish_reason(raw: str | None) -> FinishReason:
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
        "model_length": FinishReason.LENGTH,
    }
    return mapping.get(raw or "", FinishReason.UNKNOWN)


def _to_mistral_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


class MistralLLMClient(LLMClient):
    """
    LLM client adapter for Mistral AI's Chat Completions API.

    Supported models:
      - ``mistral-large-latest``    — most capable
      - ``mistral-small-latest``    — fast and efficient
      - ``open-mistral-nemo``       — open-weight, 128K context
      - ``codestral-latest``        — code generation specialist

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``MISTRAL_API_KEY`` environment variable

    Example::

        from conduit_sdk.providers.mistral import MistralLLMClient
        from conduit_sdk.core.config import ClientConfig

        client = MistralLLMClient(ClientConfig(
            provider="mistral",
            model="mistral-large-latest",
            api_key="...",   # or set MISTRAL_API_KEY env var
        ))
        response = await client.generate(
            LLMRequest.Builder()
            .system("Be concise.")
            .user("Explain mixture-of-experts.")
            .max_tokens(200)
            .build()
        )
        print(response.content)
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("MISTRAL_API_KEY", "")

    def _sdk_client(self):
        mistralai = _require_mistral()
        return mistralai.Mistral(
            api_key=self._api_key(),
            timeout_ms=int(self.config.timeout_seconds * 1000),
        )

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": _to_mistral_messages(request.messages),
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        kwargs.update(request.extra)
        return kwargs

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        try:
            raw = await self._sdk_client().chat.complete_async(**self._build_kwargs(request))
        except Exception as exc:
            raise _wrap_mistral_error(exc) from exc

        choice = raw.choices[0]
        content = choice.message.content or ""
        if isinstance(content, list):  # content blocks
            content = " ".join(
                c.text if hasattr(c, "text") else str(c) for c in content
            )

        finish_raw = (
            str(choice.finish_reason).split(".")[-1].lower() if choice.finish_reason else None
        )

        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=content),
            finish_reason=_map_finish_reason(finish_raw),
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
                completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
                total_tokens=raw.usage.total_tokens if raw.usage else 0,
            ),
            model=raw.model or (request.model or self.config.model),
            provider="mistral",
            raw_response=raw,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        try:
            stream = await self._sdk_client().chat.stream_async(**self._build_kwargs(request))
        except Exception as exc:
            raise _wrap_mistral_error(exc) from exc

        async for event in stream:
            delta = event.data.choices[0].delta.content if event.data.choices else None
            if delta:
                yield delta
