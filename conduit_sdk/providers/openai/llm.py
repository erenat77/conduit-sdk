"""
OpenAILLMClient — adapter for OpenAI Chat Completions API.

Supports:
  - generate()  : non-streaming chat completions
  - stream()    : token-by-token streaming via AsyncStream
  - Tool / function calling via request.tools
  - Per-request model override via request.model
  - Automatic cost population when CostConfig pricing is set
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
from conduit_sdk.models.responses import FinishReason, LLMResponse, ToolCall

if TYPE_CHECKING:
    import openai


def _require_openai() -> openai:
    try:
        import openai  # noqa: PLC0415

        return openai
    except ImportError:
        raise ImportError(
            "OpenAI provider requires the openai package. "
            "Install it with: pip install llm-conduit[openai]"
        ) from None


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for m in messages:
        entry: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.name:
            entry["name"] = m.name
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        result.append(entry)
    return result


def _map_finish_reason(raw: str | None) -> FinishReason:
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(raw or "", FinishReason.UNKNOWN)


def _wrap_openai_error(exc: Exception, provider: str = "openai") -> ProviderError:
    """Convert openai SDK exceptions to conduit_sdk exceptions."""
    openai = _require_openai()

    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(str(exc), provider=provider)
    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(str(exc), provider=provider)
    if isinstance(exc, openai.APITimeoutError):
        return TimeoutError(str(exc), provider=provider)
    if isinstance(exc, openai.APIStatusError):
        return ProviderError(
            str(exc),
            provider=provider,
            status_code=exc.status_code,
            raw_response=exc.response,
        )
    return ProviderError(str(exc), provider=provider)


class OpenAILLMClient(LLMClient):
    """
    LLM client adapter for OpenAI's Chat Completions API.

    Supported models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1-*, o3-*, …

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``OPENAI_API_KEY`` environment variable

    Parameters
    ----------
    config:
        ``ClientConfig`` with at minimum ``model`` set.  Set ``api_key`` here
        or via the ``OPENAI_API_KEY`` environment variable.

    Example::

        client = OpenAILLMClient(ClientConfig(
            model="gpt-4o",
            api_key="sk-...",
            cost=CostConfig(
                input_cost_per_1k_tokens=0.005,
                output_cost_per_1k_tokens=0.015,
            ),
        ))
        response = await client.generate(LLMRequest(
            messages=[Message.user("What is the speed of light?")],
            max_tokens=256,
            temperature=0.3,
        ))
        print(response.content)
        print(f"Tokens: {response.usage.total_tokens}")
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("OPENAI_API_KEY", "")

    def _sdk_client(self) -> openai.AsyncOpenAI:
        openai = _require_openai()
        return openai.AsyncOpenAI(
            api_key=self._api_key(),
            base_url=self.config.api_base_url or None,
            timeout=self.config.timeout_seconds,
        )

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": _to_openai_messages(request.messages),
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop"] = request.stop
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        kwargs.update(request.extra)
        return kwargs

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        import json

        try:
            raw = await self._sdk_client().chat.completions.create(**self._build_kwargs(request))
        except Exception as exc:
            raise _wrap_openai_error(exc) from exc

        choice = raw.choices[0]

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=choice.message.content or "",
            ),
            finish_reason=_map_finish_reason(choice.finish_reason),
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
                completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
                total_tokens=raw.usage.total_tokens if raw.usage else 0,
            ),
            model=raw.model,
            provider="openai",
            raw_response=raw,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True

        try:
            stream = await self._sdk_client().chat.completions.create(**kwargs)
        except Exception as exc:
            raise _wrap_openai_error(exc) from exc

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
