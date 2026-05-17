"""
Request models — typed inputs for each modality.

All request models are immutable Pydantic models (frozen=True).
Provider-specific fields go in ``extra`` to keep the core models stable.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from conduit_sdk.models.common import Message


class _BaseRequest(BaseModel):
    """Common fields shared by all requests."""

    model_config = ConfigDict(frozen=True)

    model: str | None = Field(
        default=None,
        description="Override the client-level model for this request.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific parameters not covered by the base schema.",
    )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """JSON-schema definition of a callable tool (function calling)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


class LLMRequest(_BaseRequest):
    """
    Request for a large-language-model chat completion.

    Parameters
    ----------
    messages:
        Conversation history.  Must contain at least one message.
    max_tokens:
        Upper bound on completion length.
    temperature:
        Sampling temperature [0, 2].  Lower = more deterministic.
    top_p:
        Nucleus sampling probability mass.
    stop:
        Up to 4 stop sequences.
    tools:
        Tool definitions for function-calling.
    stream:
        Hint to the client to use the streaming endpoint.
        ``LLMClient.stream()`` always streams regardless of this flag.
    """

    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = Field(default=None, max_length=4)
    tools: list[ToolDefinition] | None = None
    stream: bool = False


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class ImageSize(BaseModel):
    model_config = ConfigDict(frozen=True)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


class ImageGenRequest(_BaseRequest):
    """
    Request for text-to-image or image-to-image generation.

    Parameters
    ----------
    prompt:
        Text description of the desired image.
    negative_prompt:
        Concepts to exclude from the generated image.
    reference_image_url:
        Seed image for image-to-image tasks.
    size:
        Output dimensions.  Defaults to 1024×1024.
    num_images:
        Number of images to generate in one call.
    steps:
        Number of diffusion steps.
    guidance_scale:
        Classifier-free guidance strength.
    seed:
        Fixed seed for reproducibility.
    output_format:
        Desired image format ("png", "jpeg", "webp").
    """

    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    reference_image_url: str | None = None
    size: ImageSize = Field(default_factory=lambda: ImageSize(width=1024, height=1024))
    num_images: int = Field(default=1, ge=1, le=10)
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0.0)
    seed: int | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


class VideoGenRequest(_BaseRequest):
    """
    Request for text-to-video or image-to-video generation.

    Parameters
    ----------
    prompt:
        Text description of the desired video.
    reference_image_url:
        First frame / keyframe for image-to-video tasks.
    duration_seconds:
        Requested clip length.
    fps:
        Frames per second.
    resolution:
        Output video dimensions.
    seed:
        Fixed seed for reproducibility.
    """

    prompt: str = Field(min_length=1)
    reference_image_url: str | None = None
    duration_seconds: float = Field(default=4.0, gt=0.0, le=300.0)
    fps: int = Field(default=24, ge=1, le=120)
    resolution: ImageSize = Field(default_factory=lambda: ImageSize(width=1280, height=720))
    seed: int | None = None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingRequest(_BaseRequest):
    """
    Request to encode one or more inputs into dense vector embeddings.

    Parameters
    ----------
    inputs:
        List of strings (or base-64 encoded images, depending on provider).
    dimensions:
        Optional target dimensionality (for models that support MRL).
    encoding_format:
        "float" (default) or "base64".
    input_type:
        Hint for asymmetric retrieval ("query" vs "document").
    """

    inputs: list[str] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] = "float"
    input_type: Literal["query", "document", "image"] | None = None


# ---------------------------------------------------------------------------
# Union type alias used by middleware
# ---------------------------------------------------------------------------

AnyRequest = Annotated[
    LLMRequest | ImageGenRequest | VideoGenRequest | EmbeddingRequest,
    Field(discriminator=None),
]
