"""
Tests for the LLMRequestBuilder fluent builder.

All tests are unit tests — no external dependencies required.
"""

from __future__ import annotations

import pytest

from conduit_sdk.models.common import MessageRole
from conduit_sdk.models.requests import LLMRequest, LLMRequestBuilder, ToolDefinition


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def test_build_classmethod_returns_builder() -> None:
    builder = LLMRequest.Builder()
    assert isinstance(builder, LLMRequestBuilder)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def test_system_message() -> None:
    req = LLMRequest.Builder().system("Be concise.").user("Hello").build()
    assert req.messages[0].role == MessageRole.SYSTEM
    assert req.messages[0].content == "Be concise."


def test_user_message() -> None:
    req = LLMRequest.Builder().user("Hello there.").build()
    assert req.messages[0].role == MessageRole.USER
    assert req.messages[0].content == "Hello there."


def test_assistant_message() -> None:
    req = LLMRequest.Builder().user("Hi").assistant("Hello!").user("How are you?").build()
    assert req.messages[1].role == MessageRole.ASSISTANT
    assert req.messages[1].content == "Hello!"


def test_multi_turn_conversation_order() -> None:
    req = (
        LLMRequest.Builder()
        .system("You are helpful.")
        .user("What is 2+2?")
        .assistant("4.")
        .user("And 3+3?")
        .build()
    )
    assert len(req.messages) == 4
    assert req.messages[0].role == MessageRole.SYSTEM
    assert req.messages[1].role == MessageRole.USER
    assert req.messages[2].role == MessageRole.ASSISTANT
    assert req.messages[3].role == MessageRole.USER
    assert req.messages[3].content == "And 3+3?"


def test_explicit_message_role() -> None:
    req = LLMRequest.Builder().message("user", "ping").build()
    assert req.messages[0].role == MessageRole.USER
    assert req.messages[0].content == "ping"


def test_message_role_accepts_enum() -> None:
    req = LLMRequest.Builder().message(MessageRole.USER, "ping").build()
    assert req.messages[0].role == MessageRole.USER


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------


def test_max_tokens() -> None:
    req = LLMRequest.Builder().user("Hi").max_tokens(256).build()
    assert req.max_tokens == 256


def test_temperature() -> None:
    req = LLMRequest.Builder().user("Hi").temperature(0.7).build()
    assert req.temperature == pytest.approx(0.7)


def test_top_p() -> None:
    req = LLMRequest.Builder().user("Hi").top_p(0.9).build()
    assert req.top_p == pytest.approx(0.9)


def test_stop_sequences() -> None:
    req = LLMRequest.Builder().user("Hi").stop("STOP", "END").build()
    assert req.stop == ["STOP", "END"]


def test_stop_single_sequence() -> None:
    req = LLMRequest.Builder().user("Hi").stop("\n\n").build()
    assert req.stop == ["\n\n"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_tools_attached() -> None:
    tool = ToolDefinition(
        name="get_weather",
        description="Returns current weather.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    req = LLMRequest.Builder().user("Weather in Paris?").tools(tool).build()
    assert req.tools is not None
    assert len(req.tools) == 1
    assert req.tools[0].name == "get_weather"


def test_multiple_tools() -> None:
    t1 = ToolDefinition(name="tool_a", description="A", parameters={})
    t2 = ToolDefinition(name="tool_b", description="B", parameters={})
    req = LLMRequest.Builder().user("Go").tools(t1, t2).build()
    assert req.tools is not None
    assert len(req.tools) == 2


# ---------------------------------------------------------------------------
# Request-level overrides
# ---------------------------------------------------------------------------


def test_model_override() -> None:
    req = LLMRequest.Builder().user("Hi").model("gpt-4o-mini").build()
    assert req.model == "gpt-4o-mini"


def test_stream_default_false() -> None:
    req = LLMRequest.Builder().user("Hi").build()
    assert req.stream is False


def test_stream_enabled() -> None:
    req = LLMRequest.Builder().user("Hi").stream().build()
    assert req.stream is True


def test_stream_explicit_false() -> None:
    req = LLMRequest.Builder().user("Hi").stream(False).build()
    assert req.stream is False


def test_with_extra() -> None:
    req = LLMRequest.Builder().user("Hi").with_extra(seed=42, logprobs=True).build()
    assert req.extra["seed"] == 42
    assert req.extra["logprobs"] is True


def test_with_extra_merges_multiple_calls() -> None:
    req = (
        LLMRequest.Builder()
        .user("Hi")
        .with_extra(seed=42)
        .with_extra(logprobs=True)
        .build()
    )
    assert req.extra["seed"] == 42
    assert req.extra["logprobs"] is True


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_are_none_or_false() -> None:
    req = LLMRequest.Builder().user("Hello").build()
    assert req.max_tokens is None
    assert req.temperature is None
    assert req.top_p is None
    assert req.stop is None
    assert req.tools is None
    assert req.stream is False
    assert req.model is None
    assert req.extra == {}


# ---------------------------------------------------------------------------
# Validation errors (Pydantic)
# ---------------------------------------------------------------------------


def test_build_raises_on_empty_messages() -> None:
    """build() with no messages should raise a Pydantic ValidationError."""
    from pydantic import ValidationError

    builder = LLMRequest.Builder()
    with pytest.raises(ValidationError):
        builder.build()


def test_temperature_out_of_range_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMRequest.Builder().user("Hi").temperature(3.0).build()


def test_top_p_out_of_range_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMRequest.Builder().user("Hi").top_p(1.5).build()


# ---------------------------------------------------------------------------
# Immutability of the built request
# ---------------------------------------------------------------------------


def test_built_request_is_frozen() -> None:
    req = LLMRequest.Builder().user("Hello").build()
    with pytest.raises(Exception):
        req.max_tokens = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Builder is reusable (each build() call creates a fresh LLMRequest)
# ---------------------------------------------------------------------------


def test_builder_reuse_produces_independent_requests() -> None:
    builder = LLMRequest.Builder().user("Hi").max_tokens(50)
    req1 = builder.build()
    req2 = builder.temperature(0.9).build()

    # req1 has no temperature, req2 does
    assert req1.temperature is None
    assert req2.temperature == pytest.approx(0.9)
    # Both share the same message list reference, which is fine since
    # LLMRequest is frozen and Message is frozen too.
    assert req1.max_tokens == 50
    assert req2.max_tokens == 50
