"""
Response models — typed outputs for each modality.

All response models are immutable.  Provider-specific metadata lives in
``raw_response`` so callers can inspect it without us coupling to a vendor.
"""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        pass


from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from conduit_sdk.models.common import Cost, Message, Usage


class _BaseResponse(BaseModel):
    """Common fields shared by all responses."""

    model_config = ConfigDict(frozen=True)

    model: str = ""
    provider: str = ""
    usage: Usage = Field(default_factory=Usage)
    cost: Cost | None = None
    latency_ms: float | None = None
    raw_response: Any | None = Field(default=None, repr=False)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(_BaseResponse):
    """
    Response from a large-language-model completion.

    Attributes
    ----------
    message:
        The assistant's reply.
    finish_reason:
        Why the model stopped generating.
    tool_calls:
        Parsed tool/function calls requested by the model (if any).
    """

    message: Message
    finish_reason: FinishReason = FinishReason.UNKNOWN
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def content(self) -> str:
        """Convenience shortcut to the assistant message content."""
        return self.message.content


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class GeneratedImage(BaseModel):
    """A single generated image."""

    model_config = ConfigDict(frozen=True)

    url: str | None = None
    b64_data: str | None = None  # base-64 encoded image bytes
    revised_prompt: str | None = None  # prompt rewrite from provider
    seed: int | None = None
    mime_type: str = "image/png"


class ImageGenResponse(_BaseResponse):
    """
    Response from an image-generation call.

    Attributes
    ----------
    images:
        One or more generated images.
    """

    images: list[GeneratedImage] = Field(default_factory=list)

    @property
    def first(self) -> GeneratedImage | None:
        """Convenience: return the first image, or None if empty."""
        return self.images[0] if self.images else None


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


class GeneratedVideo(BaseModel):
    """A single generated video clip."""

    model_config = ConfigDict(frozen=True)

    url: str | None = None
    duration_seconds: float | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    mime_type: str = "video/mp4"


class VideoGenResponse(_BaseResponse):
    """
    Response from a video-generation call.

    Attributes
    ----------
    videos:
        One or more generated video clips.
    """

    videos: list[GeneratedVideo] = Field(default_factory=list)

    @property
    def first(self) -> GeneratedVideo | None:
        return self.videos[0] if self.videos else None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class Embedding(BaseModel):
    """A single embedding vector."""

    model_config = ConfigDict(frozen=True)

    index: int
    vector: list[float]
    object: str = "embedding"


class EmbeddingResponse(_BaseResponse):
    """
    Response from an embedding call.

    Attributes
    ----------
    embeddings:
        Ordered list of embedding vectors (same order as inputs).
    dimensions:
        Dimensionality of each vector.
    """

    embeddings: list[Embedding] = Field(default_factory=list)

    @property
    def vectors(self) -> list[list[float]]:
        """Convenience: return just the raw float vectors."""
        return [e.vector for e in self.embeddings]

    @property
    def dimensions(self) -> int | None:
        return len(self.embeddings[0].vector) if self.embeddings else None


# ---------------------------------------------------------------------------
# Union type alias used by middleware
# ---------------------------------------------------------------------------

AnyResponse = Annotated[
    LLMResponse | ImageGenResponse | VideoGenResponse | EmbeddingResponse,
    Field(discriminator=None),
]
