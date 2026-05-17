"""
Provider skeleton — OpenAI
===========================
Covers: LLM (chat + streaming + tool calling), Embeddings.

Install: pip install openai
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from conduit_sdk.clients import EmbeddingClient, LLMClient
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import EmbeddingRequest, LLMRequest
from conduit_sdk.models.responses import (
    Embedding,
    EmbeddingResponse,
    FinishReason,
    LLMResponse,
    ToolCall,
)


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


class OpenAILLMClient(LLMClient):
    """
    LLM adapter for OpenAI's Chat Completions API.

    Usage::

        client = OpenAILLMClient(ClientConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-...",
            cost=CostConfig(
                input_cost_per_1k_tokens=0.005,
                output_cost_per_1k_tokens=0.015,
            ),
        ))
        response = await client.generate(LLMRequest(messages=[Message.user("Hi")]))
    """

    def _sdk(self):
        import openai  # noqa: PLC0415

        return openai.AsyncOpenAI(api_key=self.config.api_key)

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            messages=_to_openai_messages(request.messages),
        )
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

        raw = await self._sdk().chat.completions.create(**kwargs)
        choice = raw.choices[0]

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            import json

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
            finish_reason=FinishReason(choice.finish_reason or "unknown"),
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
            ),
            model=raw.model,
            provider="openai",
            raw_response=raw,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            messages=_to_openai_messages(request.messages),
            stream=True,
        )
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        kwargs.update(request.extra)

        async with await self._sdk().chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


class OpenAIEmbeddingClient(EmbeddingClient):
    """
    Embedding adapter for OpenAI's Embeddings API.

    Usage::

        client = OpenAIEmbeddingClient(ClientConfig(
            provider="openai",
            model="text-embedding-3-large",
            api_key="sk-...",
        ))
        response = await client.embed(EmbeddingRequest(inputs=["Hello world"]))
    """

    def _sdk(self):
        import openai  # noqa: PLC0415

        return openai.AsyncOpenAI(api_key=self.config.api_key)

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            input=request.inputs,
            encoding_format=request.encoding_format,
        )
        if request.dimensions is not None:
            kwargs["dimensions"] = request.dimensions
        kwargs.update(request.extra)

        raw = await self._sdk().embeddings.create(**kwargs)

        return EmbeddingResponse(
            embeddings=[Embedding(index=item.index, vector=item.embedding) for item in raw.data],
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                total_tokens=raw.usage.total_tokens,
                embedding_count=len(request.inputs),
            ),
            model=raw.model,
            provider="openai",
            raw_response=raw,
        )
