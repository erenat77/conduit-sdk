"""
Example 11 — FastAPI model API server
======================================
Shows how to wrap conduit-sdk behind a production-ready FastAPI service with:

  - POST /chat          — single-turn and multi-turn LLM completion
  - POST /chat/stream   — streaming completion via Server-Sent Events (SSE)
  - POST /embed         — dense vector embeddings
  - POST /imagine       — text-to-image generation
  - GET  /health        — liveness probe

Run
---
  pip install "conduit-sdk[openai]" fastapi uvicorn

  USE_MOCK=1 uvicorn examples.11_fastapi_server:app --reload

Then try:
  curl -s -X POST http://localhost:8000/chat \
       -H "Content-Type: application/json" \
       -d '{"messages": [{"role": "user", "content": "What is RLHF?"}]}'
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from conduit_sdk.clients import EmbeddingClient, ImageGenClient, LLMClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.exceptions import ModelSDKError as ConduitError
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import EmbeddingRequest, ImageGenRequest, LLMRequest
from conduit_sdk.models.responses import (
    EmbeddingResponse,
    FinishReason,
    GeneratedImage,
    ImageGenResponse,
    LLMResponse,
)


class MockLLMClient(LLMClient):
    async def _generate(self, request: LLMRequest) -> LLMResponse:
        last = next(m.content for m in reversed(request.messages) if m.role == MessageRole.USER)
        reply = f"[mock] You asked: '{last}'. Here is a concise answer."
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=reply),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=12, completion_tokens=18, total_tokens=30),
            model=self.config.model,
            provider=self.config.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        response = await self._generate(request)
        for word in response.content.split():
            yield word + " "


class MockEmbeddingClient(EmbeddingClient):
    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        from conduit_sdk.models.responses import Embedding

        return EmbeddingResponse(
            embeddings=[
                Embedding(index=i, vector=[0.1 * (i + 1)] * 8) for i in range(len(request.inputs))
            ],
            model=self.config.model,
            provider=self.config.provider,
            usage=Usage(
                prompt_tokens=len(request.inputs) * 4, total_tokens=len(request.inputs) * 4
            ),
        )


class MockImageClient(ImageGenClient):
    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        return ImageGenResponse(
            images=[
                GeneratedImage(
                    url=f"https://mock.example.com/image?prompt={request.prompt[:20]}",
                    revised_prompt=request.prompt,
                )
            ],
            model=self.config.model,
            provider=self.config.provider,
        )


USE_MOCK = os.getenv("USE_MOCK", "0") == "1"


def _build_llm_client() -> LLMClient:
    if USE_MOCK:
        return MockLLMClient(config=ClientConfig(provider="mock", model="mock-llm-v1"))
    raise RuntimeError("Set USE_MOCK=1 or wire a real provider adapter.")


def _build_embedding_client() -> EmbeddingClient:
    if USE_MOCK:
        return MockEmbeddingClient(config=ClientConfig(provider="mock", model="mock-embed-v1"))
    raise RuntimeError("Set USE_MOCK=1 or wire a real provider adapter.")


def _build_image_client() -> ImageGenClient:
    if USE_MOCK:
        return MockImageClient(config=ClientConfig(provider="mock", model="mock-image-v1"))
    raise RuntimeError("Set USE_MOCK=1 or wire a real provider adapter.")


_clients: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _clients["llm"] = _build_llm_client()
    _clients["embed"] = _build_embedding_client()
    _clients["image"] = _build_image_client()
    yield
    _clients.clear()


app = FastAPI(
    title="conduit-sdk model API",
    description="Provider-agnostic LLM / embedding / image API powered by conduit-sdk.",
    version="0.1.3",
    lifespan=lifespan,
)


def get_llm_client() -> LLMClient:
    return _clients["llm"]


def get_embedding_client() -> EmbeddingClient:
    return _clients["embed"]


def get_image_client() -> ImageGenClient:
    return _clients["image"]


class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    model: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
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


class ImagineRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = None
    width: int = 1024
    height: int = 1024
    num_images: int = Field(default=1, ge=1, le=10)
    output_format: Literal["png", "jpeg", "webp"] = "png"


class ImagineResponse(BaseModel):
    images: list[dict[str, Any]]
    model: str
    provider: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    client: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
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


@app.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    client: LLMClient = Depends(get_llm_client),
) -> StreamingResponse:
    async def _event_stream() -> AsyncIterator[str]:
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

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    body: EmbedRequest,
    client: EmbeddingClient = Depends(get_embedding_client),
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


@app.post("/imagine", response_model=ImagineResponse)
async def imagine(
    body: ImagineRequest,
    client: ImageGenClient = Depends(get_image_client),
) -> ImagineResponse:
    try:
        builder = (
            ImageGenRequest.Builder()
            .prompt(body.prompt)
            .size(body.width, body.height)
            .num_images(body.num_images)
            .output_format(body.output_format)
        )
        if body.model:
            builder.model(body.model)
        response = await client.generate(builder.build())
        return ImagineResponse(
            images=[img.model_dump(exclude_none=True) for img in response.images],
            model=response.model,
            provider=response.provider,
        )
    except ConduitError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.11_fastapi_server:app", host="0.0.0.0", port=8000, reload=True)
