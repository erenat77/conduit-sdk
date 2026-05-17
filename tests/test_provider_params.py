"""
Tests for ProviderParams — typed extension point for provider-specific fields.

Covers:
  - Subclassing ProviderParams and mixing into a base request
  - Extended builder returning the custom type
  - Overriding Request.Builder to use the extended builder
  - Frozen immutability on extended requests
  - model_dump() round-trip for extended requests
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field, ValidationError

from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import (
    EmbeddingRequest,
    ImageGenRequest,
    LLMRequest,
    LLMRequestBuilder,
    ProviderParams,
)

# ---------------------------------------------------------------------------
# Shared param groups (defined once, reused across tests)
# ---------------------------------------------------------------------------


class OpenAIParams(ProviderParams):
    """Hypothetical OpenAI-specific parameters."""

    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    parallel_tool_calls: bool = True
    logprobs: bool = False


class AnthropicParams(ProviderParams):
    """Hypothetical Anthropic-specific parameters."""

    thinking_budget_tokens: int = 1024
    betas: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extended request classes
# ---------------------------------------------------------------------------


class OpenAILLMRequest(LLMRequest, OpenAIParams):
    """LLMRequest extended with OpenAI-specific parameters."""

    pass


class AnthropicLLMRequest(LLMRequest, AnthropicParams):
    """LLMRequest extended with Anthropic-specific parameters."""

    pass


class OpenAIImageRequest(ImageGenRequest, OpenAIParams):
    """ImageGenRequest extended with OpenAI-specific parameters."""

    pass


class OpenAIEmbeddingRequest(EmbeddingRequest, OpenAIParams):
    """EmbeddingRequest extended with OpenAI-specific parameters."""

    pass


# ---------------------------------------------------------------------------
# Extended builder
# ---------------------------------------------------------------------------


class OpenAILLMRequestBuilder(LLMRequestBuilder):
    """LLMRequestBuilder that produces OpenAILLMRequest instances."""

    def __init__(self) -> None:
        super().__init__()
        self._reasoning_effort: str = "medium"
        self._parallel_tool_calls: bool = True
        self._logprobs: bool = False

    def reasoning_effort(self, value: str) -> OpenAILLMRequestBuilder:
        self._reasoning_effort = value
        return self

    def parallel_tool_calls(self, enabled: bool) -> OpenAILLMRequestBuilder:
        self._parallel_tool_calls = enabled
        return self

    def logprobs(self, enabled: bool = True) -> OpenAILLMRequestBuilder:
        self._logprobs = enabled
        return self

    def build(self) -> OpenAILLMRequest:  # type: ignore[override]
        base = super().build()
        return OpenAILLMRequest(
            **base.model_dump(),
            reasoning_effort=self._reasoning_effort,
            parallel_tool_calls=self._parallel_tool_calls,
            logprobs=self._logprobs,
        )


# Wire the extended builder onto the extended request class
OpenAILLMRequest.Builder = OpenAILLMRequestBuilder


# ---------------------------------------------------------------------------
# ProviderParams unit tests
# ---------------------------------------------------------------------------


class TestProviderParams:
    def test_subclass_is_frozen(self) -> None:
        params = OpenAIParams()
        with pytest.raises(ValidationError):
            params.reasoning_effort = "low"  # type: ignore[misc]

    def test_defaults_applied(self) -> None:
        params = OpenAIParams()
        assert params.reasoning_effort == "medium"
        assert params.parallel_tool_calls is True
        assert params.logprobs is False

    def test_custom_values(self) -> None:
        params = OpenAIParams(reasoning_effort="high", parallel_tool_calls=False)
        assert params.reasoning_effort == "high"
        assert params.parallel_tool_calls is False

    def test_list_field_default_factory(self) -> None:
        a = AnthropicParams()
        b = AnthropicParams()
        # Each instance gets its own list — no shared mutable default
        assert a.betas is not b.betas


# ---------------------------------------------------------------------------
# Extended request construction tests
# ---------------------------------------------------------------------------


class TestExtendedRequestConstruction:
    def test_direct_construction_with_base_fields_only(self) -> None:
        req = OpenAILLMRequest(messages=[Message.user("Hi")])
        assert req.reasoning_effort == "medium"
        assert req.parallel_tool_calls is True

    def test_direct_construction_with_all_fields(self) -> None:
        req = OpenAILLMRequest(
            messages=[Message.user("Hi")],
            max_tokens=100,
            reasoning_effort="high",
            parallel_tool_calls=False,
        )
        assert req.max_tokens == 100
        assert req.reasoning_effort == "high"
        assert req.parallel_tool_calls is False

    def test_extended_request_is_frozen(self) -> None:
        req = OpenAILLMRequest(messages=[Message.user("Hi")])
        with pytest.raises(ValidationError):
            req.reasoning_effort = "low"  # type: ignore[misc]

    def test_is_instance_of_base_request(self) -> None:
        req = OpenAILLMRequest(messages=[Message.user("Hi")])
        assert isinstance(req, LLMRequest)
        assert isinstance(req, OpenAIParams)

    def test_model_dump_includes_extra_fields(self) -> None:
        req = OpenAILLMRequest(messages=[Message.user("Hi")], reasoning_effort="low")
        dumped = req.model_dump()
        assert dumped["reasoning_effort"] == "low"
        assert "messages" in dumped

    def test_anthropic_params_mixin(self) -> None:
        req = AnthropicLLMRequest(
            messages=[Message.user("Hi")],
            thinking_budget_tokens=4096,
            betas=["interleaved-thinking-2025-05-14"],
        )
        assert req.thinking_budget_tokens == 4096
        assert req.betas == ["interleaved-thinking-2025-05-14"]

    def test_params_mix_into_image_request(self) -> None:
        req = OpenAIImageRequest(
            prompt="A cyberpunk city",
            reasoning_effort="high",
        )
        assert isinstance(req, ImageGenRequest)
        assert req.reasoning_effort == "high"
        assert req.prompt == "A cyberpunk city"

    def test_params_mix_into_embedding_request(self) -> None:
        req = OpenAIEmbeddingRequest(
            inputs=["Hello world"],
            logprobs=True,
        )
        assert isinstance(req, EmbeddingRequest)
        assert req.logprobs is True


# ---------------------------------------------------------------------------
# Extended builder tests
# ---------------------------------------------------------------------------


class TestExtendedBuilder:
    def test_builder_returns_extended_type(self) -> None:
        req = (
            OpenAILLMRequestBuilder()
            .user("Hi")
            .build()
        )
        assert isinstance(req, OpenAILLMRequest)
        assert isinstance(req, LLMRequest)

    def test_builder_sets_base_fields(self) -> None:
        req = (
            OpenAILLMRequestBuilder()
            .system("Be concise.")
            .user("Explain RLHF.")
            .max_tokens(200)
            .temperature(0.4)
            .build()
        )
        assert len(req.messages) == 2
        assert req.max_tokens == 200
        assert req.temperature == pytest.approx(0.4)

    def test_builder_sets_extended_fields(self) -> None:
        req = (
            OpenAILLMRequestBuilder()
            .user("Hi")
            .reasoning_effort("high")
            .parallel_tool_calls(False)
            .logprobs(True)
            .build()
        )
        assert req.reasoning_effort == "high"
        assert req.parallel_tool_calls is False
        assert req.logprobs is True

    def test_builder_defaults_for_extended_fields(self) -> None:
        req = OpenAILLMRequestBuilder().user("Hi").build()
        assert req.reasoning_effort == "medium"
        assert req.parallel_tool_calls is True
        assert req.logprobs is False

    def test_full_chain(self) -> None:
        req = (
            OpenAILLMRequestBuilder()
            .system("You are a reasoning assistant.")
            .user("Explain chain-of-thought prompting.")
            .max_tokens(500)
            .temperature(0.2)
            .reasoning_effort("high")
            .parallel_tool_calls(False)
            .logprobs(True)
            .model("o1-preview")
            .with_extra(seed=42)
            .build()
        )
        assert req.reasoning_effort == "high"
        assert req.model == "o1-preview"
        assert req.extra["seed"] == 42
        assert isinstance(req, OpenAILLMRequest)


# ---------------------------------------------------------------------------
# Builder override via Request.Builder
# ---------------------------------------------------------------------------


class TestBuilderClassOverride:
    def test_override_builder_on_extended_request(self) -> None:
        """
        After `OpenAILLMRequest.Builder = OpenAILLMRequestBuilder`,
        callers can use the standard `Request.Builder()` API.
        """
        req = (
            OpenAILLMRequest.Builder()
            .system("Be direct.")
            .user("What is attention in transformers?")
            .reasoning_effort("high")
            .build()
        )
        assert isinstance(req, OpenAILLMRequest)
        assert req.reasoning_effort == "high"

    def test_base_request_builder_still_returns_base_type(self) -> None:
        """
        Overriding the Builder on a subclass must not affect the base class.
        """
        req = LLMRequest.Builder().user("Hi").build()
        assert type(req) is LLMRequest
        assert not isinstance(req, OpenAILLMRequest)
