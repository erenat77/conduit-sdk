"""
Shared fixtures and helpers for provider tests.

Unit tests use mocked SDK objects — no API key required.
Integration tests are marked @pytest.mark.integration and only run
when the relevant API key (OPENAI_API_KEY / ANTHROPIC_API_KEY) is set.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.core.middleware import MiddlewarePipeline

# ──────────────────────────────────────────────────────────────
# Auto-skip integration tests on known billing / quota errors
# ──────────────────────────────────────────────────────────────
_SKIP_MESSAGES = (
    "credit balance is too low",
    "insufficient_quota",
    "exceeded your current quota",
    "billing_hard_limit_reached",
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore[return]
    """
    Convert integration test failures caused by billing/quota errors into skips.
    pytest-asyncio in AUTO mode doesn't propagate async exceptions through sync
    fixtures, so we hook at the report level instead.
    """
    outcome = yield
    rep = outcome.get_result()

    if rep.when != "call" or not rep.failed:
        return
    if not item.get_closest_marker("integration"):
        return

    excinfo = call.excinfo
    if excinfo is None:
        return

    msg = str(excinfo.value).lower()
    if any(phrase in msg for phrase in _SKIP_MESSAGES):
        rep.outcome = "skipped"
        rep.longrepr = (
            str(item.fspath),
            item.location[1],
            f"Skipped: billing limit reached — {excinfo.value}",
        )
        raise


def openai_config(model: str = "gpt-4o") -> ClientConfig:
    return ClientConfig(
        provider="openai",
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY", "sk-test-key"),
        cost=CostConfig(
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.015,
        ),
    )


def anthropic_config(model: str = "claude-haiku-4-5-20251001") -> ClientConfig:
    return ClientConfig(
        provider="anthropic",
        model=model,
        api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-test-key"),
        cost=CostConfig(
            input_cost_per_1k_tokens=0.00025,
            output_cost_per_1k_tokens=0.00125,
        ),
    )


def bare_pipeline() -> MiddlewarePipeline:
    """Empty pipeline — skips retry/rate-limit so unit tests run fast."""
    return MiddlewarePipeline([])


# ── Integration skip guard ──────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: calls real provider APIs — set OPENAI_API_KEY / ANTHROPIC_API_KEY to run",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests when the required API key is not set."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    skip_openai = pytest.mark.skip(
        reason="OPENAI_API_KEY not set — skipping OpenAI integration tests"
    )
    skip_anthropic = pytest.mark.skip(
        reason="ANTHROPIC_API_KEY not set — skipping Anthropic integration tests"
    )

    for item in items:
        if not item.get_closest_marker("integration"):
            continue
        # Determine provider from module path
        module_path = str(getattr(item, "fspath", ""))
        if "anthropic" in module_path:
            if not anthropic_key:
                item.add_marker(skip_anthropic)
        else:
            if not openai_key:
                item.add_marker(skip_openai)


# ── OpenAI SDK mock builders ────────────────────────────────────────────────


def make_chat_completion(
    content: str = "Hello from mock!",
    model: str = "gpt-4o",
    prompt_tokens: int = 20,
    completion_tokens: int = 30,
    finish_reason: str = "stop",
):
    """Build a minimal mock of openai.types.chat.ChatCompletion."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    completion.model = model
    return completion


def make_embedding_response(
    vectors: list[list[float]],
    model: str = "text-embedding-3-large",
    prompt_tokens: int = 10,
):
    """Build a minimal mock of openai.types.CreateEmbeddingResponse."""
    items = []
    for i, vec in enumerate(vectors):
        item = MagicMock()
        item.index = i
        item.embedding = vec
        items.append(item)

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.total_tokens = prompt_tokens

    resp = MagicMock()
    resp.data = items
    resp.usage = usage
    resp.model = model
    return resp


def make_image_response(urls: list[str], revised_prompt: str | None = None):
    """Build a minimal mock of openai.types.ImagesResponse."""
    items = []
    for url in urls:
        item = MagicMock()
        item.url = url
        item.revised_prompt = revised_prompt
        items.append(item)

    resp = MagicMock()
    resp.data = items
    return resp


async def aiter_chunks(texts: list[str]):
    """Async generator that yields mock OpenAI stream chunks."""
    for text in texts:
        delta = MagicMock()
        delta.content = text
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        yield chunk


# ── Anthropic SDK mock builders ─────────────────────────────────────────────


def make_anthropic_text_block(text: str = "Hello from Anthropic mock!") -> MagicMock:
    """Create a mock Anthropic TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_anthropic_tool_use_block(
    id: str = "toolu_01ABC",
    name: str = "get_weather",
    input: dict | None = None,
) -> MagicMock:
    """Create a mock Anthropic ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input or {"location": "Paris"}
    return block


def make_anthropic_message(
    content: str = "Hello from Anthropic mock!",
    model: str = "claude-haiku-4-5-20251001",
    input_tokens: int = 20,
    output_tokens: int = 30,
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Build a minimal mock of anthropic.types.Message (non-streaming)."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    msg = MagicMock()
    msg.content = [make_anthropic_text_block(content)]
    msg.stop_reason = stop_reason
    msg.usage = usage
    msg.model = model
    return msg


def make_anthropic_tool_message(
    tool_id: str = "toolu_01ABC",
    tool_name: str = "get_weather",
    tool_input: dict | None = None,
    model: str = "claude-haiku-4-5-20251001",
    input_tokens: int = 25,
    output_tokens: int = 15,
) -> MagicMock:
    """Build a mock Anthropic Message containing a tool_use block."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    msg = MagicMock()
    msg.content = [
        make_anthropic_tool_use_block(
            id=tool_id,
            name=tool_name,
            input=tool_input or {"location": "Paris"},
        )
    ]
    msg.stop_reason = "tool_use"
    msg.usage = usage
    msg.model = model
    return msg


def make_anthropic_stream_context(texts: list[str]) -> MagicMock:
    """
    Build a mock async context manager returned by `client.messages.stream()`.
    The `.text_stream` attribute is an async iterator over the given texts.
    """

    async def _text_stream():
        for t in texts:
            yield t

    stream = MagicMock()
    stream.text_stream = _text_stream()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=stream)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm
