"""
CohereLLMClient — adapter for Cohere Chat API v2.

Uses the official ``cohere`` Python SDK (v2 client: ``cohere.AsyncClientV2``).

Supported models: command-r-plus-08-2024, command-r-08-2024,
                  command-a-03-2025, …

Install: pip install llm-conduit[cohere]
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


def _require_cohere():
    try:
        import cohere  # noqa: PLC0415

        return cohere
    except ImportError:
        raise ImportError(
            "Cohere provider requires the cohere package. "
            "Install it with: pip install llm-conduit[cohere]"
        ) from None


def _wrap_cohere_error(exc: Exception, provider: str = "cohere") -> ProviderError:
    exc_type = type(exc).__name__
    if "Unauthorized" in exc_type or "Authentication" in exc_type or "403" in str(exc):
        return AuthenticationError(str(exc), provider=provider)
    if "RateLimit" in exc_type or "TooManyRequests" in exc_type or "429" in str(exc):
        return RateLimitError(str(exc), provider=provider)
    if "Timeout" in exc_type or "Deadline" in exc_type:
        return TimeoutError(str(exc), provider=provider)
    if hasattr(exc, "status_code"):
        return ProviderError(str(exc), provider=provider, status_code=exc.status_code)
    return ProviderError(str(exc), provider=provider)


def _map_finish_reason(raw: str | None) -> FinishReason:
    mapping = {
        "COMPLETE": FinishReason.STOP,
        "MAX_TOKENS": FinishReason.LENGTH,
        "STOP_SEQUENCE": FinishReason.STOP,
        "TOOL_CALL": FinishReason.TOOL_CALLS,
        "ERROR": FinishReason.UNKNOWN,
        "ERROR_TOXIC": FinishReason.CONTENT_FILTER,
        "ERROR_LIMIT": FinishReason.LENGTH,
        "USER_CANCEL": FinishReason.UNKNOWN,
    }
    return mapping.get(raw or "", FinishReason.UNKNOWN)


def _to_cohere_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """
    Convert conduit messages to Cohere v2 chat message format.

    Cohere v2 roles: 'system' | 'user' | 'assistant' | 'tool'
    """
    result = []
    for m in messages:
        role = m.role.value  # system / user / assistant
        result.append({"role": role, "content": m.content})
    return result


class CohereLLMClient(LLMClient):
    """
    LLM client adapter for Cohere's Chat API (v2).

    Supported models:
      - ``command-r-plus-08-2024``   — most capable, RAG-optimised
      - ``command-r-08-2024``        — balanced speed/quality
      - ``command-a-03-2025``        — latest flagship
      - ``command-light``            — fastest / cheapest

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``COHERE_API_KEY`` environment variable

    Example::

        from conduit_sdk.providers.cohere import CohereLLMClient
        from conduit_sdk.core.config import ClientConfig

        client = CohereLLMClient(ClientConfig(
            provider="cohere",
            model="command-r-plus-08-2024",
            api_key="...",   # or set COHERE_API_KEY env var
        ))
        response = await client.generate(
            LLMRequest.Builder()
            .system("Be concise.")
            .user("What is RAG?")
            .max_tokens(200)
            .build()
        )
        print(response.content)
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("COHERE_API_KEY", "")

    def _sdk_client(self):
        cohere = _require_cohere()
        return cohere.AsyncClientV2(
            api_key=self._api_key(),
            timeout=self.config.timeout_seconds,
        )

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": _to_cohere_messages(request.messages),
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["p"] = request.top_p  # Cohere uses 'p' for nucleus sampling
        if request.stop:
            kwargs["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        kwargs.update(request.extra)
        return kwargs

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        try:
            raw = await self._sdk_client().chat(**self._build_kwargs(request))
        except Exception as exc:
            raise _wrap_cohere_error(exc) from exc

        # v2 response: raw.message.content is a list of content blocks
        content_blocks = raw.message.content if raw.message else []
        content = ""
        if content_blocks:
            content = " ".join(b.text if hasattr(b, "text") else str(b) for b in content_blocks)

        finish_raw = str(raw.finish_reason).split(".")[-1] if raw.finish_reason else None

        usage = raw.usage
        billed = usage.billed_units if usage else None
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=content),
            finish_reason=_map_finish_reason(finish_raw),
            usage=Usage(
                prompt_tokens=billed.input_tokens if billed else 0,
                completion_tokens=billed.output_tokens if billed else 0,
                total_tokens=(
                    (billed.input_tokens or 0) + (billed.output_tokens or 0) if billed else 0
                ),
            ),
            model=request.model or self.config.model,
            provider="cohere",
            raw_response=raw,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        try:
            async for event in self._sdk_client().chat_stream(**self._build_kwargs(request)):
                event_type = type(event).__name__
                if "ContentDelta" in event_type or "TextGeneration" in event_type:
                    delta = getattr(event, "text", None) or (
                        event.delta.message.content.text
                        if hasattr(event, "delta")
                        and hasattr(event.delta, "message")
                        and event.delta.message
                        and event.delta.message.content
                        else None
                    )
                    if delta:
                        yield delta
        except Exception as exc:
            raise _wrap_cohere_error(exc) from exc
