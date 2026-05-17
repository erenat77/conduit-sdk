"""
Shared fixtures and mock provider implementations used across all tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from conduit_sdk.clients.embedding import EmbeddingClient
from conduit_sdk.clients.image import ImageGenClient
from conduit_sdk.clients.llm import LLMClient
from conduit_sdk.clients.video import VideoGenClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import (
    EmbeddingRequest,
    ImageGenRequest,
    LLMRequest,
    VideoGenRequest,
)
from conduit_sdk.models.responses import (
    Embedding,
    EmbeddingResponse,
    FinishReason,
    GeneratedImage,
    GeneratedVideo,
    ImageGenResponse,
    LLMResponse,
    VideoGenResponse,
)

# ---------------------------------------------------------------------------
# Mock clients — minimal concrete implementations for testing
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """Echo client: returns the last user message as the assistant reply."""

    call_count = 0

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        content = request.messages[-1].content
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=f"Echo: {content}"),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            model=request.model or self.model,
            provider=self.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        content = request.messages[-1].content
        for word in content.split():
            yield word + " "


class MockImageGenClient(ImageGenClient):
    """Returns a single placeholder image."""

    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        return ImageGenResponse(
            images=[GeneratedImage(url="https://example.com/image.png", seed=42)],
            usage=Usage(image_count=request.num_images),
            model=request.model or self.model,
            provider=self.provider,
        )


class MockVideoGenClient(VideoGenClient):
    """Returns a single placeholder video."""

    async def _generate(self, request: VideoGenRequest) -> VideoGenResponse:
        return VideoGenResponse(
            videos=[
                GeneratedVideo(
                    url="https://example.com/video.mp4",
                    duration_seconds=request.duration_seconds,
                    fps=request.fps,
                )
            ],
            usage=Usage(video_seconds=request.duration_seconds),
            model=request.model or self.model,
            provider=self.provider,
        )


class MockEmbeddingClient(EmbeddingClient):
    """Returns trivial unit vectors."""

    DIMS = 4

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        embeddings = [
            Embedding(index=i, vector=[1.0] * self.DIMS) for i in range(len(request.inputs))
        ]
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=Usage(
                prompt_tokens=len(request.inputs) * 5,
                total_tokens=len(request.inputs) * 5,
                embedding_count=len(request.inputs),
            ),
            model=request.model or self.model,
            provider=self.provider,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config() -> ClientConfig:
    return ClientConfig(provider="mock", model="mock-v1")


@pytest.fixture
def bare_pipeline() -> MiddlewarePipeline:
    """An empty pipeline — no retry, rate-limit, logging, or cost."""
    return MiddlewarePipeline([])


@pytest.fixture
def llm_client(base_config: ClientConfig, bare_pipeline: MiddlewarePipeline) -> MockLLMClient:
    return MockLLMClient(config=base_config, middleware=bare_pipeline)


@pytest.fixture
def image_client(
    base_config: ClientConfig, bare_pipeline: MiddlewarePipeline
) -> MockImageGenClient:
    return MockImageGenClient(config=base_config, middleware=bare_pipeline)


@pytest.fixture
def video_client(
    base_config: ClientConfig, bare_pipeline: MiddlewarePipeline
) -> MockVideoGenClient:
    return MockVideoGenClient(config=base_config, middleware=bare_pipeline)


@pytest.fixture
def embedding_client(
    base_config: ClientConfig, bare_pipeline: MiddlewarePipeline
) -> MockEmbeddingClient:
    return MockEmbeddingClient(config=base_config, middleware=bare_pipeline)
