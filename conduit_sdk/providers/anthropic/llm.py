"""
AnthropicLLMClient — adapter for Anthropic's Messages API.

Supports:
  - generate()   : non-streaming messages (claude-opus-4-*, claude-sonnet-4-*, claude-haiku-4-*)
  - stream()     : token-by-token streaming via AsyncMessageStream
  - System prompt: extracted automatically from request.messages (role=system)
  - Tool use     : request.tools → Anthropic tool_use blocks, parsed back to ToolCall
  - Per-request model override via request.model
  - Full error mapping: overloaded → RateLimitError, permission → AuthenticationError, etc.

Anthropic-specific notes
-------------------------
- System messages must be passed as the top-level ``system`` parameter, not
  inside the messages list. This adapter handles that split automatically.
- The Messages API does not return token usage on streaming; usage is only
  available on non-streaming responses.
- ``top_p`` and ``temperature`` cannot both be set at the same time per the
  Anthropic API contract; if both are present, only ``temperature`` is sent.
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
    import anthropic


def _require_anthropic() -> anthropic:
    try:
        import anthropic  # noqa: PLC0415

        return anthropic
    except ImportError:
        raise ImportError(
            "Anthropic provider requires the anthropic package. "
            "Install it with: pip install llm-conduit[anthropic]"
        ) from None


def _split_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Anthropic separates the system prompt from the conversation.
    Collects all system messages into one string, returns the rest as-is.
    """
    system_parts: list[str] = []
    user_messages: list[dict[str, Any]] = []

    for m in messages:
        if m.role == MessageRole.SYSTEM:
            system_parts.append(m.content)
        else:
            entry: dict[str, Any] = {"role": m.role.value, "content": m.content}
            user_messages.append(entry)

    system = "\n\n".join(system_parts) if system_parts else None
    return system, user_messages


def _map_finish_reason(stop_reason: str | None) -> FinishReason:
    mapping = {
        "end_turn": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "tool_use": FinishReason.TOOL_CALLS,
        "stop_sequence": FinishReason.STOP,
    }
    return mapping.get(stop_reason or "", FinishReason.UNKNOWN)


def _wrap_anthropic_error(exc: Exception) -> ProviderError:
    """Convert anthropic SDK exceptions into conduit_sdk exceptions."""
    ant = _require_anthropic()

    if isinstance(exc, ant.RateLimitError):
        return RateLimitError(str(exc), provider="anthropic")
    if isinstance(exc, ant.AuthenticationError):
        return AuthenticationError(str(exc), provider="anthropic")
    if isinstance(exc, ant.APITimeoutError):
        return TimeoutError(str(exc), provider="anthropic")
    if isinstance(exc, ant.APIStatusError):
        return ProviderError(
            str(exc),
            provider="anthropic",
            status_code=exc.status_code,
            raw_response=getattr(exc, "response", None),
        )
    return ProviderError(str(exc), provider="anthropic")


def _parse_tool_calls(content_blocks: list[Any]) -> tuple[str, list[ToolCall]]:
    """
    Extract text content and tool_use blocks from Anthropic's response content list.
    Returns the concatenated text and a list of ToolCall objects.
    """
    import json

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                    if isinstance(block.input, dict)
                    else json.loads(block.input),
                )
            )

    return "".join(text_parts), tool_calls


class AnthropicLLMClient(LLMClient):
    """
    LLM client adapter for Anthropic's Messages API.

    Supported models (as of 2026):
      - claude-opus-4-6         — most capable
      - claude-sonnet-4-6       — balanced speed and quality
      - claude-haiku-4-5-20251001 — fastest and cheapest

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``ANTHROPIC_API_KEY`` environment variable

    Parameters
    ----------
    config:
        ``ClientConfig`` with at minimum ``model`` set.

    Example::

        client = AnthropicLLMClient(ClientConfig(
            model="claude-sonnet-4-6",
            api_key="sk-ant-...",
            cost=CostConfig(
                input_cost_per_1k_tokens=0.003,
                output_cost_per_1k_tokens=0.015,
            ),
        ))

        # Non-streaming
        response = await client.generate(LLMRequest(
            messages=[
                Message.system("Be concise."),
                Message.user("Explain async/await in Python."),
            ],
            max_tokens=300,
            temperature=0.5,
        ))

        # Streaming
        async for chunk in client.stream(request):
            print(chunk, end="", flush=True)
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def _sdk_client(self) -> anthropic.AsyncAnthropic:
        ant = _require_anthropic()
        return ant.AsyncAnthropic(
            api_key=self._api_key(),
            base_url=self.config.api_base_url or None,
            timeout=self.config.timeout_seconds,
        )

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        system, messages = _split_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }

        if system:
            kwargs["system"] = system

        # Anthropic: temperature and top_p are mutually exclusive
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        elif request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop:
            kwargs["stop_sequences"] = request.stop

        if request.tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]

        kwargs.update(request.extra)
        return kwargs

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        try:
            raw = await self._sdk_client().messages.create(**self._build_kwargs(request))
        except Exception as exc:
            raise _wrap_anthropic_error(exc) from exc

        text, tool_calls = _parse_tool_calls(raw.content)

        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=text),
            finish_reason=_map_finish_reason(raw.stop_reason),
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
            ),
            model=raw.model,
            provider="anthropic",
            raw_response=raw,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        try:
            async with self._sdk_client().messages.stream(**self._build_kwargs(request)) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk
        except Exception as exc:
            raise _wrap_anthropic_error(exc) from exc
