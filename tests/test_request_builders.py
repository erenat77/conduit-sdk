"""
Tests for ImageGenRequestBuilder, VideoGenRequestBuilder, and EmbeddingRequestBuilder.

All tests are unit tests — no external dependencies required.
"""

from __future__ import annotations

import pytest

from conduit_sdk.models.requests import (
    EmbeddingRequest,
    EmbeddingRequestBuilder,
    ImageGenRequest,
    ImageGenRequestBuilder,
    VideoGenRequest,
    VideoGenRequestBuilder,
)


# ===========================================================================
# ImageGenRequestBuilder
# ===========================================================================


class TestImageGenRequestBuilder:
    def test_build_classmethod_returns_builder(self) -> None:
        assert isinstance(ImageGenRequest.Builder(), ImageGenRequestBuilder)

    def test_prompt(self) -> None:
        req = ImageGenRequest.Builder().prompt("A fox in snow").build()
        assert req.prompt == "A fox in snow"

    def test_negative_prompt(self) -> None:
        req = ImageGenRequest.Builder().prompt("Forest").negative_prompt("blurry").build()
        assert req.negative_prompt == "blurry"

    def test_reference_image_url(self) -> None:
        url = "https://example.com/img.jpg"
        req = ImageGenRequest.Builder().prompt("Style transfer").reference_image_url(url).build()
        assert req.reference_image_url == url

    def test_size(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").size(512, 768).build()
        assert req.size.width == 512
        assert req.size.height == 768

    def test_num_images(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").num_images(4).build()
        assert req.num_images == 4

    def test_steps(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").steps(50).build()
        assert req.steps == 50

    def test_guidance_scale(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").guidance_scale(7.5).build()
        assert req.guidance_scale == pytest.approx(7.5)

    def test_seed(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").seed(42).build()
        assert req.seed == 42

    def test_output_format_jpeg(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").output_format("jpeg").build()
        assert req.output_format == "jpeg"

    def test_model_override(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").model("dall-e-3").build()
        assert req.model == "dall-e-3"

    def test_with_extra(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").with_extra(quality="hd").build()
        assert req.extra["quality"] == "hd"

    def test_defaults(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").build()
        assert req.negative_prompt is None
        assert req.reference_image_url is None
        assert req.size.width == 1024
        assert req.size.height == 1024
        assert req.num_images == 1
        assert req.steps is None
        assert req.guidance_scale is None
        assert req.seed is None
        assert req.output_format == "png"
        assert req.model is None
        assert req.extra == {}

    def test_built_request_is_frozen(self) -> None:
        req = ImageGenRequest.Builder().prompt("X").build()
        with pytest.raises(Exception):
            req.prompt = "Y"  # type: ignore[misc]

    def test_build_raises_on_empty_prompt(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImageGenRequest.Builder().build()  # prompt="" fails min_length=1

    def test_chaining_is_fluent(self) -> None:
        req = (
            ImageGenRequest.Builder()
            .prompt("Cyberpunk city")
            .negative_prompt("blurry")
            .size(1024, 1024)
            .num_images(2)
            .steps(30)
            .guidance_scale(8.0)
            .seed(7)
            .output_format("webp")
            .model("dall-e-3")
            .with_extra(style="vivid")
            .build()
        )
        assert req.prompt == "Cyberpunk city"
        assert req.num_images == 2
        assert req.output_format == "webp"
        assert req.extra["style"] == "vivid"


# ===========================================================================
# VideoGenRequestBuilder
# ===========================================================================


class TestVideoGenRequestBuilder:
    def test_build_classmethod_returns_builder(self) -> None:
        assert isinstance(VideoGenRequest.Builder(), VideoGenRequestBuilder)

    def test_prompt(self) -> None:
        req = VideoGenRequest.Builder().prompt("Timelapse of a flower").build()
        assert req.prompt == "Timelapse of a flower"

    def test_reference_image_url(self) -> None:
        url = "https://example.com/frame.jpg"
        req = VideoGenRequest.Builder().prompt("X").reference_image_url(url).build()
        assert req.reference_image_url == url

    def test_duration(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").duration(8.0).build()
        assert req.duration_seconds == pytest.approx(8.0)

    def test_fps(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").fps(30).build()
        assert req.fps == 30

    def test_resolution(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").resolution(1920, 1080).build()
        assert req.resolution.width == 1920
        assert req.resolution.height == 1080

    def test_seed(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").seed(99).build()
        assert req.seed == 99

    def test_model_override(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").model("sora-v1").build()
        assert req.model == "sora-v1"

    def test_with_extra(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").with_extra(motion="high").build()
        assert req.extra["motion"] == "high"

    def test_defaults(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").build()
        assert req.reference_image_url is None
        assert req.duration_seconds == pytest.approx(4.0)
        assert req.fps == 24
        assert req.resolution.width == 1280
        assert req.resolution.height == 720
        assert req.seed is None
        assert req.model is None
        assert req.extra == {}

    def test_built_request_is_frozen(self) -> None:
        req = VideoGenRequest.Builder().prompt("X").build()
        with pytest.raises(Exception):
            req.prompt = "Y"  # type: ignore[misc]

    def test_build_raises_on_empty_prompt(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoGenRequest.Builder().build()

    def test_chaining_is_fluent(self) -> None:
        req = (
            VideoGenRequest.Builder()
            .prompt("Ocean waves")
            .reference_image_url("https://example.com/ocean.jpg")
            .duration(6.0)
            .fps(24)
            .resolution(1280, 720)
            .seed(3)
            .model("runway-gen3")
            .with_extra(camera_motion="zoom_in")
            .build()
        )
        assert req.prompt == "Ocean waves"
        assert req.duration_seconds == pytest.approx(6.0)
        assert req.extra["camera_motion"] == "zoom_in"


# ===========================================================================
# EmbeddingRequestBuilder
# ===========================================================================


class TestEmbeddingRequestBuilder:
    def test_build_classmethod_returns_builder(self) -> None:
        assert isinstance(EmbeddingRequest.Builder(), EmbeddingRequestBuilder)

    def test_single_input(self) -> None:
        req = EmbeddingRequest.Builder().inputs("Hello world").build()
        assert req.inputs == ["Hello world"]

    def test_multiple_inputs(self) -> None:
        req = EmbeddingRequest.Builder().inputs("foo", "bar", "baz").build()
        assert req.inputs == ["foo", "bar", "baz"]

    def test_add_input(self) -> None:
        req = (
            EmbeddingRequest.Builder()
            .inputs("first")
            .add_input("second")
            .add_input("third")
            .build()
        )
        assert req.inputs == ["first", "second", "third"]

    def test_dimensions(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").dimensions(1536).build()
        assert req.dimensions == 1536

    def test_encoding_format_base64(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").encoding_format("base64").build()
        assert req.encoding_format == "base64"

    def test_input_type_query(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").input_type("query").build()
        assert req.input_type == "query"

    def test_input_type_document(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").input_type("document").build()
        assert req.input_type == "document"

    def test_model_override(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").model("text-embedding-3-large").build()
        assert req.model == "text-embedding-3-large"

    def test_with_extra(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").with_extra(user="u123").build()
        assert req.extra["user"] == "u123"

    def test_defaults(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").build()
        assert req.dimensions is None
        assert req.encoding_format == "float"
        assert req.input_type is None
        assert req.model is None
        assert req.extra == {}

    def test_built_request_is_frozen(self) -> None:
        req = EmbeddingRequest.Builder().inputs("x").build()
        with pytest.raises(Exception):
            req.inputs = ["y"]  # type: ignore[misc]

    def test_build_raises_on_empty_inputs(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EmbeddingRequest.Builder().build()  # inputs=[] fails min_length=1

    def test_chaining_is_fluent(self) -> None:
        req = (
            EmbeddingRequest.Builder()
            .inputs("How does RLHF work?", "Explain transformers")
            .dimensions(1536)
            .encoding_format("float")
            .input_type("query")
            .model("text-embedding-3-large")
            .with_extra(user="u42")
            .build()
        )
        assert len(req.inputs) == 2
        assert req.dimensions == 1536
        assert req.input_type == "query"
        assert req.extra["user"] == "u42"
