"""
Request models — typed inputs for each modality.

All request models are immutable Pydantic models (frozen=True).

Builder pattern
---------------
Every request class exposes a ``Builder`` for ergonomic construction::

    request = (
        LLMRequest.Builder()
        .system("Be concise.")
        .user("Explain async/await in Python.")
        .max_tokens(200)
        .temperature(0.3)
        .build()
    )

Every builder inherits from :class:`RequestBuilder` and can be subclassed
to add provider-specific fields::

    class o1Builder(LLMRequestBuilder):
        def __init__(self) -> None:
            super().__init__()
            self._reasoning_effort = "medium"

        def reasoning_effort(self, v: str) -> o1Builder:
            self._reasoning_effort = v
            return self

        def build(self) -> LLMRequest:
            base = super().build()
            return base.model_copy(update={"extra": {**base.extra,
                                                     "reasoning_effort": self._reasoning_effort}})
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from conduit_sdk.models.common import Message, MessageRole

# ===========================================================================
# Shared bases
# ===========================================================================


class ProviderParams(BaseModel):
    """
    Base class for typed provider-specific parameter groups.

    Subclass and mix into any request class via multiple inheritance.
    Every field **must** have a default value.

    Example::

        class OpenAIParams(ProviderParams):
            reasoning_effort: str = "medium"

        class OpenAILLMRequest(LLMRequest, OpenAIParams):
            pass
    """

    model_config = ConfigDict(frozen=True)


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


class ToolDefinition(BaseModel):
    """JSON-schema definition of a callable tool (function calling)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]


# ===========================================================================
# Base builder
# ===========================================================================


class RequestBuilder:
    """
    Base class for all request builders.

    Provides the two fields every request shares — ``model`` and ``extra`` —
    and defines the ``build()`` contract that subclasses must fulfil.

    Subclass one of the four concrete builders to extend it::

        class o1Builder(LLMRequestBuilder):
            def __init__(self) -> None:
                super().__init__()
                self._reasoning_effort = "medium"

            def reasoning_effort(self, v: str) -> o1Builder:
                self._reasoning_effort = v
                return self

            def build(self) -> LLMRequest:
                return super().build().model_copy(
                    update={"extra": {"reasoning_effort": self._reasoning_effort}}
                )

        o1Request.Builder = o1Builder
    """

    def __init__(self) -> None:
        self._model: str | None = None
        self._extra: dict[str, Any] = {}

    def model(self, model_name: str) -> RequestBuilder:
        """Override the client-level model for this request."""
        self._model = model_name
        return self

    def with_extra(self, **kwargs: Any) -> RequestBuilder:
        """Pass provider-specific parameters not covered by the base schema."""
        self._extra.update(kwargs)
        return self

    def build(self) -> Any:
        """Construct and return the immutable request. Must be overridden."""
        raise NotImplementedError("RequestBuilder subclasses must implement build()")


# ===========================================================================
# Request classes  (declared before builders — builders reference them)
# ===========================================================================


class LLMRequest(_BaseRequest):
    """
    Request for a chat-completion call.

    Examples
    --------
    ::

        request = (
            LLMRequest.Builder()
            .system("Be concise.")
            .user("What is a transformer?")
            .max_tokens(150)
            .build()
        )
    """

    Builder: ClassVar[Any] = None  # wired to LLMRequestBuilder below

    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = Field(default=None, max_length=4)
    tools: list[ToolDefinition] | None = None
    stream: bool = False


class ImageSize(BaseModel):
    model_config = ConfigDict(frozen=True)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


class ImageGenRequest(_BaseRequest):
    """
    Request for text-to-image or image-to-image generation.

    Examples
    --------
    ::

        request = (
            ImageGenRequest.Builder()
            .prompt("A fox in a snowy forest at dawn")
            .size(1024, 1024)
            .num_images(2)
            .build()
        )
    """

    Builder: ClassVar[Any] = None  # wired to ImageGenRequestBuilder below

    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    reference_image_url: str | None = None
    size: ImageSize = Field(default_factory=lambda: ImageSize(width=1024, height=1024))
    num_images: int = Field(default=1, ge=1, le=10)
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0.0)
    seed: int | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"


class VideoGenRequest(_BaseRequest):
    """
    Request for text-to-video or image-to-video generation.

    Examples
    --------
    ::

        request = (
            VideoGenRequest.Builder()
            .prompt("A timelapse of a blooming flower")
            .duration(8.0)
            .fps(30)
            .build()
        )
    """

    Builder: ClassVar[Any] = None  # wired to VideoGenRequestBuilder below

    prompt: str = Field(min_length=1)
    reference_image_url: str | None = None
    duration_seconds: float = Field(default=4.0, gt=0.0, le=300.0)
    fps: int = Field(default=24, ge=1, le=120)
    resolution: ImageSize = Field(default_factory=lambda: ImageSize(width=1280, height=720))
    seed: int | None = None


class EmbeddingRequest(_BaseRequest):
    """
    Request for dense vector embeddings.

    Examples
    --------
    ::

        request = (
            EmbeddingRequest.Builder()
            .inputs("How does RLHF work?", "What is a transformer?")
            .dimensions(1536)
            .input_type("query")
            .build()
        )
    """

    Builder: ClassVar[Any] = None  # wired to EmbeddingRequestBuilder below

    inputs: list[str] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] = "float"
    input_type: Literal["query", "document", "image"] | None = None


# ===========================================================================
# Concrete builders  (one per modality, all extend RequestBuilder)
# ===========================================================================


class LLMRequestBuilder(RequestBuilder):
    """
    Fluent builder for :class:`LLMRequest`.

    Use via ``LLMRequest.Builder()`` or subclass to extend::

        class o1Builder(LLMRequestBuilder):
            def reasoning_effort(self, v): ...
    """

    def __init__(self) -> None:
        super().__init__()
        self._messages: list[Message] = []
        self._max_tokens: int | None = None
        self._temperature: float | None = None
        self._top_p: float | None = None
        self._stop: list[str] | None = None
        self._tools: list[ToolDefinition] | None = None
        self._stream: bool = False

    def system(self, content: str) -> LLMRequestBuilder:
        self._messages.append(Message(role=MessageRole.SYSTEM, content=content))
        return self

    def user(self, content: str) -> LLMRequestBuilder:
        self._messages.append(Message(role=MessageRole.USER, content=content))
        return self

    def assistant(self, content: str) -> LLMRequestBuilder:
        self._messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        return self

    def message(self, role: str | MessageRole, content: str) -> LLMRequestBuilder:
        self._messages.append(Message(role=MessageRole(role), content=content))
        return self

    def max_tokens(self, value: int) -> LLMRequestBuilder:
        self._max_tokens = value
        return self

    def temperature(self, value: float) -> LLMRequestBuilder:
        self._temperature = value
        return self

    def top_p(self, value: float) -> LLMRequestBuilder:
        self._top_p = value
        return self

    def stop(self, *sequences: str) -> LLMRequestBuilder:
        self._stop = list(sequences)
        return self

    def tools(self, *tool_defs: ToolDefinition) -> LLMRequestBuilder:
        self._tools = list(tool_defs)
        return self

    def stream(self, enabled: bool = True) -> LLMRequestBuilder:
        self._stream = enabled
        return self

    def model(self, model_name: str) -> LLMRequestBuilder:  # type: ignore[override]
        self._model = model_name
        return self

    def with_extra(self, **kwargs: Any) -> LLMRequestBuilder:  # type: ignore[override]
        self._extra.update(kwargs)
        return self

    def build(self) -> LLMRequest:
        return LLMRequest(
            messages=self._messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            stop=self._stop,
            tools=self._tools,
            stream=self._stream,
            model=self._model,
            extra=self._extra,
        )


LLMRequest.Builder = LLMRequestBuilder


class ImageGenRequestBuilder(RequestBuilder):
    """
    Fluent builder for :class:`ImageGenRequest`.

    Use via ``ImageGenRequest.Builder()`` or subclass to extend.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prompt: str = ""
        self._negative_prompt: str | None = None
        self._reference_image_url: str | None = None
        self._size: ImageSize = ImageSize(width=1024, height=1024)
        self._num_images: int = 1
        self._steps: int | None = None
        self._guidance_scale: float | None = None
        self._seed: int | None = None
        self._output_format: Literal["png", "jpeg", "webp"] = "png"

    def prompt(self, text: str) -> ImageGenRequestBuilder:
        self._prompt = text
        return self

    def negative_prompt(self, text: str) -> ImageGenRequestBuilder:
        self._negative_prompt = text
        return self

    def reference_image_url(self, url: str) -> ImageGenRequestBuilder:
        self._reference_image_url = url
        return self

    def size(self, width: int, height: int) -> ImageGenRequestBuilder:
        self._size = ImageSize(width=width, height=height)
        return self

    def num_images(self, count: int) -> ImageGenRequestBuilder:
        self._num_images = count
        return self

    def steps(self, value: int) -> ImageGenRequestBuilder:
        self._steps = value
        return self

    def guidance_scale(self, value: float) -> ImageGenRequestBuilder:
        self._guidance_scale = value
        return self

    def seed(self, value: int) -> ImageGenRequestBuilder:
        self._seed = value
        return self

    def output_format(self, fmt: Literal["png", "jpeg", "webp"]) -> ImageGenRequestBuilder:
        self._output_format = fmt
        return self

    def model(self, model_name: str) -> ImageGenRequestBuilder:  # type: ignore[override]
        self._model = model_name
        return self

    def with_extra(self, **kwargs: Any) -> ImageGenRequestBuilder:  # type: ignore[override]
        self._extra.update(kwargs)
        return self

    def build(self) -> ImageGenRequest:
        return ImageGenRequest(
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            reference_image_url=self._reference_image_url,
            size=self._size,
            num_images=self._num_images,
            steps=self._steps,
            guidance_scale=self._guidance_scale,
            seed=self._seed,
            output_format=self._output_format,
            model=self._model,
            extra=self._extra,
        )


ImageGenRequest.Builder = ImageGenRequestBuilder


class VideoGenRequestBuilder(RequestBuilder):
    """
    Fluent builder for :class:`VideoGenRequest`.

    Use via ``VideoGenRequest.Builder()`` or subclass to extend.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prompt: str = ""
        self._reference_image_url: str | None = None
        self._duration_seconds: float = 4.0
        self._fps: int = 24
        self._resolution: ImageSize = ImageSize(width=1280, height=720)
        self._seed: int | None = None

    def prompt(self, text: str) -> VideoGenRequestBuilder:
        self._prompt = text
        return self

    def reference_image_url(self, url: str) -> VideoGenRequestBuilder:
        self._reference_image_url = url
        return self

    def duration(self, seconds: float) -> VideoGenRequestBuilder:
        self._duration_seconds = seconds
        return self

    def fps(self, value: int) -> VideoGenRequestBuilder:
        self._fps = value
        return self

    def resolution(self, width: int, height: int) -> VideoGenRequestBuilder:
        self._resolution = ImageSize(width=width, height=height)
        return self

    def seed(self, value: int) -> VideoGenRequestBuilder:
        self._seed = value
        return self

    def model(self, model_name: str) -> VideoGenRequestBuilder:  # type: ignore[override]
        self._model = model_name
        return self

    def with_extra(self, **kwargs: Any) -> VideoGenRequestBuilder:  # type: ignore[override]
        self._extra.update(kwargs)
        return self

    def build(self) -> VideoGenRequest:
        return VideoGenRequest(
            prompt=self._prompt,
            reference_image_url=self._reference_image_url,
            duration_seconds=self._duration_seconds,
            fps=self._fps,
            resolution=self._resolution,
            seed=self._seed,
            model=self._model,
            extra=self._extra,
        )


VideoGenRequest.Builder = VideoGenRequestBuilder


class EmbeddingRequestBuilder(RequestBuilder):
    """
    Fluent builder for :class:`EmbeddingRequest`.

    Use via ``EmbeddingRequest.Builder()`` or subclass to extend.
    """

    def __init__(self) -> None:
        super().__init__()
        self._inputs: list[str] = []
        self._dimensions: int | None = None
        self._encoding_format: Literal["float", "base64"] = "float"
        self._input_type: Literal["query", "document", "image"] | None = None

    def inputs(self, *texts: str) -> EmbeddingRequestBuilder:
        self._inputs = list(texts)
        return self

    def add_input(self, text: str) -> EmbeddingRequestBuilder:
        self._inputs.append(text)
        return self

    def dimensions(self, value: int) -> EmbeddingRequestBuilder:
        self._dimensions = value
        return self

    def encoding_format(self, fmt: Literal["float", "base64"]) -> EmbeddingRequestBuilder:
        self._encoding_format = fmt
        return self

    def input_type(self, kind: Literal["query", "document", "image"]) -> EmbeddingRequestBuilder:
        self._input_type = kind
        return self

    def model(self, model_name: str) -> EmbeddingRequestBuilder:  # type: ignore[override]
        self._model = model_name
        return self

    def with_extra(self, **kwargs: Any) -> EmbeddingRequestBuilder:  # type: ignore[override]
        self._extra.update(kwargs)
        return self

    def build(self) -> EmbeddingRequest:
        return EmbeddingRequest(
            inputs=self._inputs,
            dimensions=self._dimensions,
            encoding_format=self._encoding_format,
            input_type=self._input_type,
            model=self._model,
            extra=self._extra,
        )


EmbeddingRequest.Builder = EmbeddingRequestBuilder


# ===========================================================================
# Union type alias used by middleware
# ===========================================================================

AnyRequest = Annotated[
    LLMRequest | ImageGenRequest | VideoGenRequest | EmbeddingRequest,
    Field(discriminator=None),
]
