"""
GroqLLMClient — adapter for Groq's Chat Completions API.

Groq is OpenAI API-compatible, so the wire format is identical.
The main differentiator is the ``groq`` SDK and the ``GROQ_API_KEY``
environment variable.

Supported models: llama-3.3-70b-versatile, llama-3.1-8b-instant,
                  mixtral-8x7b-32768, gemma2-9b-it, …

Install: pip install llm-conduit[groq]
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
    import groq as groq_sdk


def _require_groq() -> groq_sdk:
    try:
        import groq  # noqa: PLC0415

        return groq
    except ImportError:
        raise ImportError(
            "Groq provider requires the groq package. "
            "Install it with: pip install llm-conduit[groq]"
        ) from None


def _wrap_groq_error(exc: Exception, provider: str = "groq") -> ProviderError:
    groq = _require_groq()

    if isinstance(exc, groq.RateLimitError):
        return RateLimitError(str(exc), provider=provider)
    if isinstance(exc, groq.AuthenticationError):
        return AuthenticationError(str(exc), provider=provider)
    if isinstance(exc, groq.APITimeoutError):
        return TimeoutError(str(exc), provider=provider)
    if isinstance(exc, groq.APIStatusError):
        return ProviderError(
            str(exc),
            provider=provider,
            status_code=exc.status_code,
        )
    return ProviderError(str(exc), provider=provider)


def _map_finish_reason(raw: str | None) -> FinishReason:
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(raw or "", FinishReason.UNKNOWN)


def _to_groq_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


class GroqLLMClient(LLMClient):
    """
    LLM client adapter for Groq's Chat Completions API.

    Groq provides ultra-fast inference via custom LPU hardware.
    The API is OpenAI-compatible, so the request/response shape is identical.

    Supported models:
      - ``llama-3.3-70b-versatile``   — best quality
      - ``llama-3.1-8b-instant``      — fastest / cheapest
      - ``mixtral-8x7b-32768``        — 32K context
      - ``gemma2-9b-it``              — Google Gemma

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``GROQ_API_KEY`` environment variable

    Example::

        from conduit_sdk.providers.groq import GroqLLMClient
        from conduit_sdk.core.config import ClientConfig

        client = GroqLLMClient(ClientConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key="gsk_...",   # or set GROQ_API_KEY env var
        ))
        response = await client.generate(
            LLMRequest.Builder().user("Explain transformers in one paragraph.").build()
        )
        print(response.content)
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("GROQ_API_KEY", "")

    def _sdk_client(self):
        groq = _require_groq()
        return groq.AsyncGroq(
            api_key=self._api_key(),
            timeout=self.config.timeout_seconds,
        )

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": _to_groq_messages(request.messages),
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop"] = request.stop
        kwargs.update(request.extra)
        return kwargs

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        try:
            raw = await self._sdk_client().chat.completions.create(**self._build_kwargs(request))
        except Exception as exc:
            raise _wrap_groq_error(exc) from exc

        choice = raw.choices[0]
        return LLMResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=choice.message.content or "",
            ),
            finish_reason=_map_finish_reason(choice.finish_reason),
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
                completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
                total_tokens=raw.usage.total_tokens if raw.usage else 0,
            ),
            model=raw.model,
            provider="groq",
            raw_response=raw,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True

        try:
            stream = await self._sdk_client().chat.completions.create(**kwargs)
        except Exception as exc:
            raise _wrap_groq_error(exc) from exc

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
