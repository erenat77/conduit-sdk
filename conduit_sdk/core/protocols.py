"""
Structural protocols for each model modality.

Design
------
Uses ``typing.Protocol`` (PEP 544) so third-party clients that happen to
implement the right interface are compatible *without* inheriting from our
abstract base classes — true duck typing with static-analysis support.

Each protocol exposes only the public surface a consumer needs; internal
lifecycle hooks live on the ABC side (``core/base.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from conduit_sdk.models.requests import (
    EmbeddingRequest,
    ImageGenRequest,
    LLMRequest,
    VideoGenRequest,
)
from conduit_sdk.models.responses import (
    EmbeddingResponse,
    ImageGenResponse,
    LLMResponse,
    VideoGenResponse,
)


@runtime_checkable
class LLMProtocol(Protocol):
    """Structural protocol for large-language-model clients."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Send a chat/completion request and return a complete response."""
        ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Send a streaming request and yield text delta chunks.

        Usage::

            async for chunk in client.stream(request):
                print(chunk, end="", flush=True)
        """
        ...

    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper around ``generate``."""
        ...


@runtime_checkable
class ImageGenProtocol(Protocol):
    """Structural protocol for image-generation clients."""

    async def generate(self, request: ImageGenRequest) -> ImageGenResponse:
        """Generate one or more images from a text prompt (or image input)."""
        ...

    def generate_sync(self, request: ImageGenRequest) -> ImageGenResponse:
        """Synchronous wrapper around ``generate``."""
        ...


@runtime_checkable
class VideoGenProtocol(Protocol):
    """Structural protocol for video-generation clients."""

    async def generate(self, request: VideoGenRequest) -> VideoGenResponse:
        """Generate a video clip from a prompt or reference image."""
        ...

    def generate_sync(self, request: VideoGenRequest) -> VideoGenResponse:
        """Synchronous wrapper around ``generate``."""
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Structural protocol for text/multi-modal embedding clients."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Encode inputs into a dense vector representation."""
        ...

    def embed_sync(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Synchronous wrapper around ``embed``."""
        ...
