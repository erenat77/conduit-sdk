"""
Example 12 — FastAPI server with real async providers (OpenAI / Anthropic)
===========================================================================
Key improvements over example 11:
  - Real async provider adapters with shared httpx connection pool
  - HTTP client created ONCE at startup, closed cleanly on shutdown
  - Provider selected via PROVIDER env var ("openai" | "anthropic")
  - X-API-Key auth header on all model routes
  - /v1/chat/completions and /v1/embeddings (OpenAI-compatible paths)

Install:
  pip install "conduit-sdk[openai]" anthropic fastapi uvicorn

Run:
  PROVIDER=openai OPENAI_API_KEY=sk-... API_KEY=secret \
    uvicorn examples.12_fastapi_real_providers:app --reload

  PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... API_KEY=secret \
    uvicorn examples.12_fastapi_real_providers:app --reload
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from conduit_sdk.clients import EmbeddingClient, LLMClient
from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.core.exceptions import ModelSDKError as ConduitError
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import EmbeddingRequest, LLMRequest
from conduit_sdk.models.responses import (
    Embedding,
    EmbeddingResponse,
    FinishReason,
    LLMResponse,
    ToolCall,
)


class OpenAILLMClient(LLMClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        import openai

        self._openai = openai.AsyncOpenAI(api_key=config.api_key)

    async def close(self) -> None:
        await self._openai.close()

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        import json

        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            messages=[{"role": m.role.value, "content": m.content} for m in request.messages],
        )
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop"] = request.stop
        kwargs.update(request.extra)
        raw = await self._openai.chat.completions.create(**kwargs)
        choice = raw.choices[0]
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments)
                )
                for tc in choice.message.tool_calls
            ]
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=choice.message.content or ""),
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
            messages=[{"role": m.role.value, "content": m.content} for m in request.messages],
            stream=True,
        )
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        kwargs.update(request.extra)
        async with await self._openai.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        import openai

        self._openai = openai.AsyncOpenAI(api_key=config.api_key)

    async def close(self) -> None:
        await self._openai.close()

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            input=request.inputs,
            encoding_format=request.encoding_format,
        )
        if request.dimensions is not None:
            kwargs["dimensions"] = request.dimensions
        kwargs.update(request.extra)
        raw = await self._openai.embeddings.create(**kwargs)
        return EmbeddingResponse(
            embeddings=[Embedding(index=item.index, vector=item.embedding) for item in raw.data],
            usage=Usage(prompt_tokens=raw.usage.prompt_tokens, total_tokens=raw.usage.total_tokens),
            model=raw.model,
            provider="openai",
            raw_response=raw,
        )


class AnthropicLLMClient(LLMClient):
    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        import anthropic

        self._anthropic = anthropic.AsyncAnthropic(api_key=config.api_key)

    async def close(self) -> None:
        await self._anthropic.close()

    def _split(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system = next((m.content for m in messages if m.role == MessageRole.SYSTEM), None)
        rest = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != MessageRole.SYSTEM
        ]
        return system, rest

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        system, msgs = self._split(request.messages)
        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            messages=msgs,
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
        raw = await self._anthropic.messages.create(**kwargs)
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
        system, msgs = self._split(request.messages)
        kwargs: dict[str, Any] = dict(
            model=request.model or self.config.model,
            messages=msgs,
            max_tokens=request.max_tokens or 1024,
        )
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        kwargs.update(request.extra)
        async with self._anthropic.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


PROVIDER = os.getenv("PROVIDER", "openai").lower()


def _make_llm_client() -> LLMClient:
    if PROVIDER == "openai":
        return OpenAILLMClient(
            ClientConfig(
                provider="openai",
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                cost=CostConfig(
                    input_cost_per_1k_tokens=float(os.getenv("INPUT_COST", "0.00015")),
                    output_cost_per_1k_tokens=float(os.getenv("OUTPUT_COST", "0.00060")),
                ),
            )
        )
    if PROVIDER == "anthropic":
        return AnthropicLLMClient(
            ClientConfig(
                provider="anthropic",
                model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                cost=CostConfig(
                    input_cost_per_1k_tokens=float(os.getenv("INPUT_COST", "0.00025")),
                    output_cost_per_1k_tokens=float(os.getenv("OUTPUT_COST", "0.00125")),
                ),
            )
        )
    raise RuntimeError(f"Unknown PROVIDER={PROVIDER!r}. Choose 'openai' or 'anthropic'.")


def _make_embedding_client() -> EmbeddingClient | None:
    if PROVIDER == "openai":
        return OpenAIEmbeddingClient(
            ClientConfig(
                provider="openai",
                model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        )
    return None  # Anthropic has no embedding API


_clients: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _clients["llm"] = _make_llm_client()
    _clients["embed"] = _make_embedding_client()
    yield
    for client in _clients.values():
        if client is not None and hasattr(client, "close"):
            await client.close()
    _clients.clear()


app = FastAPI(
    title="conduit-sdk real provider API",
    description="OpenAI / Anthropic — swap via PROVIDER env var.",
    version="0.1.3",
    lifespan=lifespan,
)

_API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key header."
        )


AuthDep = Depends(verify_api_key)


def get_llm() -> LLMClient:
    return _clients["llm"]


def get_embed() -> EmbeddingClient:
    client = _clients.get("embed")
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Provider '{PROVIDER}' does not support embeddings.",
        )
    return client


class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    model: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    content: str
    finish_reason: str
    model: str
    provider: str
    usage: UsageOut
    cost_usd: float | None = None


class EmbedRequest(BaseModel):
    inputs: list[str] = Field(min_length=1)
    model: str | None = None
    dimensions: int | None = Field(default=None, gt=0)
    input_type: Literal["query", "document", "image"] | None = None


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dimensions: int | None
    model: str
    provider: str
    usage: UsageOut


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": PROVIDER}


@app.post("/v1/chat/completions", response_model=ChatResponse, dependencies=[AuthDep])
async def chat(body: ChatRequest, client: LLMClient = Depends(get_llm)) -> ChatResponse:
    try:
        builder = LLMRequest.Builder()
        for m in body.messages:
            builder.message(m.role, m.content)
        if body.max_tokens:
            builder.max_tokens(body.max_tokens)
        if body.temperature is not None:
            builder.temperature(body.temperature)
        if body.top_p is not None:
            builder.top_p(body.top_p)
        if body.stop:
            builder.stop(*body.stop)
        if body.model:
            builder.model(body.model)
        response = await client.generate(builder.build())
        return ChatResponse(
            content=response.content,
            finish_reason=response.finish_reason,
            model=response.model,
            provider=response.provider,
            usage=UsageOut(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            cost_usd=response.cost.total_cost if response.cost else None,
        )
    except ConduitError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/chat/stream", dependencies=[AuthDep])
async def chat_stream(body: ChatRequest, client: LLMClient = Depends(get_llm)) -> StreamingResponse:
    async def _sse() -> AsyncIterator[str]:
        try:
            builder = LLMRequest.Builder()
            for m in body.messages:
                builder.message(m.role, m.content)
            if body.max_tokens:
                builder.max_tokens(body.max_tokens)
            if body.temperature is not None:
                builder.temperature(body.temperature)
            if body.model:
                builder.model(body.model)
            builder.stream(True)
            async for chunk in client.stream(builder.build()):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except ConduitError as exc:
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/v1/embeddings", response_model=EmbedResponse, dependencies=[AuthDep])
async def embeddings(
    body: EmbedRequest, client: EmbeddingClient = Depends(get_embed)
) -> EmbedResponse:
    try:
        builder = EmbeddingRequest.Builder().inputs(*body.inputs)
        if body.dimensions:
            builder.dimensions(body.dimensions)
        if body.input_type:
            builder.input_type(body.input_type)
        if body.model:
            builder.model(body.model)
        response = await client.embed(builder.build())
        return EmbedResponse(
            vectors=response.vectors,
            dimensions=response.dimensions,
            model=response.model,
            provider=response.provider,
            usage=UsageOut(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )
    except ConduitError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.12_fastapi_real_providers:app", host="0.0.0.0", port=8000, reload=True)
