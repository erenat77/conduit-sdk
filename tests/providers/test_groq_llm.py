"""
Unit tests for GroqLLMClient.

All tests mock the groq SDK — no API key required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.exceptions import RateLimitError
from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason
from conduit_sdk.providers.groq import GroqLLMClient
from tests.providers.conftest import bare_pipeline


def _groq_config(model: str = "llama-3.3-70b-versatile") -> ClientConfig:
    return ClientConfig(provider="groq", model=model, api_key="gsk_test")


def _client() -> GroqLLMClient:
    return GroqLLMClient(config=_groq_config(), middleware=bare_pipeline())


def _make_completion(
    content: str = "Groq mock response",
    model: str = "llama-3.3-70b-versatile",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
) -> MagicMock:
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


async def _aiter_chunks(texts: list[str]):
    for text in texts:
        delta = MagicMock()
        delta.content = text
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        yield chunk


class TestGroqLLMClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_response(self):
        mock_completion = _make_completion(content="Fast Groq answer.")
        client = _client()
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_completion)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(LLMRequest(messages=[Message.user("Hello")]))

        assert resp.content == "Fast Groq answer."
        assert resp.provider == "groq"
        assert resp.model == "llama-3.3-70b-versatile"
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self):
        mock_completion = _make_completion(prompt_tokens=8, completion_tokens=16)
        client = _client()
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_completion)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(LLMRequest(messages=[Message.user("Hi")]))

        assert resp.usage.prompt_tokens == 8
        assert resp.usage.completion_tokens == 16
        assert resp.usage.total_tokens == 24

    @pytest.mark.asyncio
    async def test_generate_passes_params(self):
        mock_completion = _make_completion()
        client = _client()
        mock_sdk = MagicMock()
        create_mock = AsyncMock(return_value=mock_completion)
        mock_sdk.chat.completions.create = create_mock
        client._sdk_client = lambda: mock_sdk

        req = LLMRequest(
            messages=[Message.user("test")],
            temperature=0.1,
            max_tokens=50,
            top_p=0.95,
        )
        await client.generate(req)

        kwargs = create_mock.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.1)
        assert kwargs["max_tokens"] == 50
        assert kwargs["top_p"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        chunks = ["Groq ", "is ", "fast!"]
        client = _client()
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=_aiter_chunks(chunks))
        client._sdk_client = lambda: mock_sdk

        req = LLMRequest(messages=[Message.user("Go!")], extra={"stream": True})
        collected = []
        async for chunk in client.stream(req):
            collected.append(chunk)

        assert collected == chunks

    @pytest.mark.asyncio
    async def test_rate_limit_error_wrapped(self):

        client = _client()

        class FakeRateLimitError(Exception):
            pass

        fake_groq = MagicMock()
        fake_groq.RateLimitError = FakeRateLimitError
        fake_groq.AuthenticationError = Exception
        fake_groq.APITimeoutError = Exception
        fake_groq.APIStatusError = Exception
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(
            side_effect=FakeRateLimitError("rate limit hit")
        )
        client._sdk_client = lambda: mock_sdk

        with (
            patch("conduit_sdk.providers.groq.llm._require_groq", return_value=fake_groq),
            pytest.raises(RateLimitError),
        ):
            await client.generate(LLMRequest(messages=[Message.user("test")]))

    def test_missing_groq_package_raises(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "groq":
                raise ImportError("No module named 'groq'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from conduit_sdk.providers.groq.llm import _require_groq

        with pytest.raises(ImportError, match="pip install llm-conduit\\[groq\\]"):
            _require_groq()

    @pytest.mark.asyncio
    async def test_model_override_per_request(self):
        mock_completion = _make_completion(model="llama-3.1-8b-instant")
        client = _client()
        mock_sdk = MagicMock()
        create_mock = AsyncMock(return_value=mock_completion)
        mock_sdk.chat.completions.create = create_mock
        client._sdk_client = lambda: mock_sdk

        req = LLMRequest(messages=[Message.user("test")], model="llama-3.1-8b-instant")
        await client.generate(req)

        assert create_mock.call_args.kwargs["model"] == "llama-3.1-8b-instant"
