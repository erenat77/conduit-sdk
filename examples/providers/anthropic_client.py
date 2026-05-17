"""
Provider skeleton — Anthropic
==============================
Covers: LLM (chat + streaming), system prompt handling.

Install: pip install anthropic
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from conduit_sdk.clients import LLMClient
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason, LLMResponse


class AnthropicLLMClient(LLMClient):
    """
    LLM adapter for Anthropic's Messages API (claude-* models).

    Usage::

        client = AnthropicLLMClient(ClientConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-ant-...",
        ))
        response = await client.generate(LLMRequest(messages=[
            Message.system("You are a helpful assistant."),
            Message.user("What is 2+2?"),
        ]))
    """

    def _sdk(self):
        import anthropic  # noqa: PLC0415

        return anthropic.AsyncAnthropic(api_key=self.config.api_key)

    def _split_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Anthropic separates system prompts from the messages list."""
        system = next(
            (m.content for m in messages if m.role == MessageRole.SYSTEM),
            None,
        )
        user_messages = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != MessageRole.SYSTEM
        ]
        return system, user_messages

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        system, user_messages = self._split_messages(request.messages)

        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            messages=user_messages,
            max_tokens=request.max_tokens or 1024,
        )
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = request.stop
        kwargs.update(request.extra)

        raw = await self._sdk().messages.create(**kwargs)

        content = raw.content[0].text if raw.content else ""
        finish = raw.stop_reason or "end_turn"

        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=content),
            finish_reason=FinishReason.STOP if finish == "end_turn" else FinishReason(finish),
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
        system, user_messages = self._split_messages(request.messages)

        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            messages=user_messages,
            max_tokens=request.max_tokens or 1024,
        )
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        kwargs.update(request.extra)

        async with self._sdk().messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
