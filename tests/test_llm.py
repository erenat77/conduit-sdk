"""Tests for LLMClient and streaming."""

from __future__ import annotations

import pytest

from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason


def _make_request(content: str = "Hello") -> LLMRequest:
    return LLMRequest(messages=[Message.user(content)])


@pytest.mark.asyncio
async def test_generate_returns_llm_response(llm_client):
    resp = await llm_client.generate(_make_request("ping"))
    assert resp.content == "Echo: ping"
    assert resp.finish_reason == FinishReason.STOP
    assert resp.usage.total_tokens == 20


@pytest.mark.asyncio
async def test_generate_sets_provider_and_model(llm_client):
    resp = await llm_client.generate(_make_request())
    assert resp.provider == "mock"
    assert resp.model == "mock-v1"


@pytest.mark.asyncio
async def test_stream_yields_words(llm_client):
    chunks: list[str] = []
    async for chunk in llm_client.stream(_make_request("hello world")):
        chunks.append(chunk)
    assert "".join(chunks).strip() == "hello world"


@pytest.mark.asyncio
async def test_generate_with_system_message(llm_client):
    req = LLMRequest(
        messages=[
            Message.system("You are a helpful assistant."),
            Message.user("What is 2+2?"),
        ]
    )
    resp = await llm_client.generate(req)
    assert "2+2" in resp.content


def test_generate_sync(llm_client):
    resp = llm_client.generate_sync(_make_request("sync test"))
    assert resp.content == "Echo: sync test"


@pytest.mark.asyncio
async def test_context_manager(base_config, bare_pipeline):
    from tests.conftest import MockLLMClient

    async with MockLLMClient(config=base_config, middleware=bare_pipeline) as client:
        resp = await client.generate(_make_request("ctx"))
    assert resp.content == "Echo: ctx"


@pytest.mark.asyncio
async def test_repr(llm_client):
    r = repr(llm_client)
    assert "MockLLMClient" in r
    assert "mock" in r
