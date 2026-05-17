"""
Tests for OpenAILLMClient.

Unit tests   — mock the openai SDK; no API key needed; always run in CI.
Integration  — real OpenAI API calls; only run when OPENAI_API_KEY is set.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.core.exceptions import AuthenticationError, RateLimitError
from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import LLMRequest, ToolDefinition
from conduit_sdk.models.responses import FinishReason
from conduit_sdk.providers.openai import OpenAILLMClient
from tests.providers.conftest import (
    aiter_chunks,
    bare_pipeline,
    make_chat_completion,
    openai_config,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _client(model: str = "gpt-4o") -> OpenAILLMClient:
    return OpenAILLMClient(config=openai_config(model), middleware=bare_pipeline())


def _req(*contents: str) -> LLMRequest:
    msgs = [Message.user(c) for c in contents]
    return LLMRequest(messages=msgs)


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS  (always run — no API key needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAILLMClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_llm_response(self):
        mock_completion = make_chat_completion(content="Paris is the capital of France.")
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(_req("What is the capital of France?"))

        assert resp.content == "Paris is the capital of France."
        assert resp.provider == "openai"
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self):
        mock_completion = make_chat_completion(prompt_tokens=15, completion_tokens=42)
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(_req("Hi"))

        assert resp.usage.prompt_tokens == 15
        assert resp.usage.completion_tokens == 42
        assert resp.usage.total_tokens == 57

    @pytest.mark.asyncio
    async def test_generate_passes_temperature_and_max_tokens(self):
        mock_completion = make_chat_completion()
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_completion)
            MockSDK.return_value.chat.completions.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            req = LLMRequest(
                messages=[Message.user("test")],
                temperature=0.2,
                max_tokens=100,
                top_p=0.9,
            )
            await client.generate(req)

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_generate_passes_stop_sequences(self):
        mock_completion = make_chat_completion()
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_completion)
            MockSDK.return_value.chat.completions.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            req = LLMRequest(messages=[Message.user("test")], stop=["END", "STOP"])
            await client.generate(req)

        assert create_mock.call_args.kwargs["stop"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_generate_with_system_message(self):
        mock_completion = make_chat_completion()
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_completion)
            MockSDK.return_value.chat.completions.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            req = LLMRequest(
                messages=[
                    Message.system("You are a helpful assistant."),
                    Message.user("Hello"),
                ]
            )
            await client.generate(req)

        messages_sent = create_mock.call_args.kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_tool_definitions(self):
        mock_completion = make_chat_completion(finish_reason="tool_calls")
        mock_completion.choices[0].message.tool_calls = []
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_completion)
            MockSDK.return_value.chat.completions.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            req = LLMRequest(
                messages=[Message.user("What is the weather in Paris?")],
                tools=[
                    ToolDefinition(
                        name="get_weather",
                        description="Returns weather for a city",
                        parameters={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    )
                ],
            )
            resp = await client.generate(req)

        assert resp.finish_reason == FinishReason.TOOL_CALLS
        sent_tools = create_mock.call_args.kwargs["tools"]
        assert sent_tools[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_generate_per_request_model_override(self):
        mock_completion = make_chat_completion(model="gpt-4-turbo")
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_completion)
            MockSDK.return_value.chat.completions.create = create_mock
            client = _client("gpt-4o")
            client._sdk_client = lambda: MockSDK.return_value

            req = LLMRequest(messages=[Message.user("test")], model="gpt-4-turbo")
            await client.generate(req)

        assert create_mock.call_args.kwargs["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        chunks = ["The ", "answer ", "is ", "42."]
        with patch("openai.AsyncOpenAI") as MockSDK:
            mock_stream = aiter_chunks(chunks)
            MockSDK.return_value.chat.completions.create = AsyncMock(return_value=mock_stream)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            collected = []
            async for chunk in client.stream(_req("What is the answer?")):
                collected.append(chunk)

        assert collected == chunks
        assert "".join(collected) == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_wraps_rate_limit_error(self):
        import openai as openai_lib

        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.chat.completions.create = AsyncMock(
                side_effect=openai_lib.RateLimitError("rate limit", response=MagicMock(), body=None)
            )
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            with pytest.raises(RateLimitError) as exc_info:
                await client.generate(_req("test"))

        assert exc_info.value.provider == "openai"

    @pytest.mark.asyncio
    async def test_generate_wraps_auth_error(self):
        import openai as openai_lib

        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.chat.completions.create = AsyncMock(
                side_effect=openai_lib.AuthenticationError(
                    "invalid key", response=MagicMock(), body=None
                )
            )
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            with pytest.raises(AuthenticationError):
                await client.generate(_req("test"))

    def test_generate_sync(self):
        mock_completion = make_chat_completion(content="Sync works!")
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value
            resp = client.generate_sync(_req("Hello"))

        assert resp.content == "Sync works!"

    def test_missing_openai_package_raises_import_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from conduit_sdk.providers.openai.llm import _require_openai

        with pytest.raises(ImportError, match="pip install conduit-sdk\\[openai\\]"):
            _require_openai()

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self):
        mock_completion = make_chat_completion()
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_completion)
            MockSDK.return_value.chat.completions.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            req = LLMRequest(
                messages=[Message.user("test")],
                extra={"seed": 42, "logprobs": True},
            )
            await client.generate(req)

        kwargs = create_mock.call_args.kwargs
        assert kwargs["seed"] == 42
        assert kwargs["logprobs"] is True


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS  (only run when OPENAI_API_KEY is set)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestOpenAILLMClientIntegration:
    """Live API tests. Skipped automatically when OPENAI_API_KEY is not set."""

    def _client(self) -> OpenAILLMClient:
        return OpenAILLMClient(
            config=ClientConfig(
                provider="openai",
                model="gpt-4o-mini",  # cheapest capable model
                cost=CostConfig(
                    input_cost_per_1k_tokens=0.00015,
                    output_cost_per_1k_tokens=0.0006,
                ),
            )
        )

    @pytest.mark.asyncio
    async def test_integration_basic_generate(self):
        client = self._client()
        req = LLMRequest(
            messages=[Message.user("Reply with exactly the word: PONG")],
            max_tokens=10,
            temperature=0.0,
        )
        resp = await client.generate(req)

        assert isinstance(resp.content, str)
        assert len(resp.content) > 0
        assert resp.usage.total_tokens > 0
        assert resp.provider == "openai"
        assert resp.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)

    @pytest.mark.asyncio
    async def test_integration_streaming(self):
        client = self._client()
        req = LLMRequest(
            messages=[Message.user("Count from 1 to 5, one number per line.")],
            max_tokens=50,
            temperature=0.0,
        )
        chunks = []
        async for chunk in client.stream(req):
            assert isinstance(chunk, str)
            chunks.append(chunk)

        full = "".join(chunks)
        assert len(full) > 0
        assert len(chunks) > 1  # must have received multiple chunks

    @pytest.mark.asyncio
    async def test_integration_system_message(self):
        client = self._client()
        req = LLMRequest(
            messages=[
                Message.system(
                    "You always respond in French, no matter what language the user uses."
                ),
                Message.user("What is the capital of France?"),
            ],
            max_tokens=50,
            temperature=0.0,
        )
        resp = await client.generate(req)
        assert resp.content  # just verify we got a response

    @pytest.mark.asyncio
    async def test_integration_multi_turn(self):
        client = self._client()
        msgs = [Message.user("My name is Eren. Remember it.")]
        resp1 = await client.generate(LLMRequest(messages=msgs, max_tokens=50, temperature=0.0))

        msgs.append(Message.assistant(resp1.content))
        msgs.append(Message.user("What is my name?"))
        resp2 = await client.generate(LLMRequest(messages=msgs, max_tokens=30, temperature=0.0))

        assert "eren" in resp2.content.lower()

    @pytest.mark.asyncio
    async def test_integration_cost_populated(self):
        config = ClientConfig(
            provider="openai",
            model="gpt-4o-mini",
            cost=CostConfig(
                input_cost_per_1k_tokens=0.00015,
                output_cost_per_1k_tokens=0.0006,
            ),
        )
        client = OpenAILLMClient(config=config)
        req = LLMRequest(messages=[Message.user("Hi")], max_tokens=5)
        resp = await client.generate(req)

        assert resp.cost is not None
        assert resp.cost.total_cost > 0

    @pytest.mark.asyncio
    async def test_integration_tool_calling(self):
        client = self._client()
        req = LLMRequest(
            messages=[Message.user("What is the weather in London? Use the get_weather tool.")],
            tools=[
                ToolDefinition(
                    name="get_weather",
                    description="Returns current weather for a given city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                )
            ],
            max_tokens=100,
            temperature=0.0,
        )
        resp = await client.generate(req)

        assert resp.finish_reason == FinishReason.TOOL_CALLS
        assert len(resp.tool_calls) > 0
        assert resp.tool_calls[0].name == "get_weather"
        assert "city" in resp.tool_calls[0].arguments
