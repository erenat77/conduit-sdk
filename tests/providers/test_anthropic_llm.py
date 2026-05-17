"""
Tests for AnthropicLLMClient (Claude models).

Unit tests   — mocked Anthropic SDK, always run (no API key needed).
Integration  — real API, requires ANTHROPIC_API_KEY env var.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.core.exceptions import AuthenticationError, RateLimitError, TimeoutError
from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason
from conduit_sdk.providers.anthropic import AnthropicLLMClient
from tests.providers.conftest import (
    anthropic_config,
    bare_pipeline,
    make_anthropic_message,
    make_anthropic_stream_context,
    make_anthropic_tool_message,
)


def _client(model: str = "claude-haiku-4-5-20251001") -> AnthropicLLMClient:
    return AnthropicLLMClient(config=anthropic_config(model), middleware=bare_pipeline())


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicLLMClientUnit:
    # ── Response mapping ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_generate_returns_message_content(self):
        mock_resp = make_anthropic_message("The speed of light is 299,792,458 m/s.")
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(
                LLMRequest(
                    messages=[Message.user("What is the speed of light?")],
                )
            )

        assert "299,792,458" in resp.content

    @pytest.mark.asyncio
    async def test_generate_provider_is_anthropic(self):
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(
                LLMRequest(
                    messages=[Message.user("Hi")],
                )
            )

        assert resp.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_generate_finish_reason_stop(self):
        mock_resp = make_anthropic_message(stop_reason="end_turn")
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(LLMRequest(messages=[Message.user("Hi")]))

        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_finish_reason_length(self):
        mock_resp = make_anthropic_message(stop_reason="max_tokens")
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(
                LLMRequest(
                    messages=[Message.user("Tell me a long story")],
                    max_tokens=5,
                )
            )

        assert resp.finish_reason == FinishReason.LENGTH

    @pytest.mark.asyncio
    async def test_generate_usage_populated(self):
        mock_resp = make_anthropic_message(input_tokens=42, output_tokens=17)
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(LLMRequest(messages=[Message.user("Hi")]))

        assert resp.usage.prompt_tokens == 42
        assert resp.usage.completion_tokens == 17
        assert resp.usage.total_tokens == 59

    @pytest.mark.asyncio
    async def test_generate_model_echoed(self):
        mock_resp = make_anthropic_message(model="claude-sonnet-4-6")
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client("claude-sonnet-4-6")
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(LLMRequest(messages=[Message.user("Hi")]))

        assert resp.model == "claude-sonnet-4-6"

    # ── System prompt extraction ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_system_prompt_extracted_from_messages(self):
        """System messages must be passed as top-level `system` param, not in messages list."""
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[
                        Message.system("You are a concise assistant."),
                        Message.user("Hi"),
                    ],
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert kwargs["system"] == "You are a concise assistant."
        # System message must NOT appear in the messages list
        roles_in_messages = [m["role"] for m in kwargs["messages"]]
        assert "system" not in roles_in_messages

    @pytest.mark.asyncio
    async def test_no_system_key_when_no_system_message(self):
        """When there's no system message, `system` kwarg must not be sent."""
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(LLMRequest(messages=[Message.user("Hi")]))

        kwargs = create_mock.call_args.kwargs
        assert "system" not in kwargs

    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(self):
        """Multiple system messages should be joined with double newlines."""
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[
                        Message.system("Be concise."),
                        Message.system("Speak like a pirate."),
                        Message.user("Hi"),
                    ],
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert "Be concise." in kwargs["system"]
        assert "Speak like a pirate." in kwargs["system"]

    # ── Temperature / top_p mutual exclusion ──────────────────────────────

    @pytest.mark.asyncio
    async def test_temperature_sent_when_set(self):
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[Message.user("Hi")],
                    temperature=0.7,
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert "top_p" not in kwargs

    @pytest.mark.asyncio
    async def test_top_p_sent_when_temperature_absent(self):
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[Message.user("Hi")],
                    top_p=0.9,
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert kwargs["top_p"] == 0.9
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_temperature_wins_over_top_p(self):
        """When both are set, only temperature is forwarded (Anthropic API contract)."""
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[Message.user("Hi")],
                    temperature=0.5,
                    top_p=0.9,
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert "temperature" in kwargs
        assert "top_p" not in kwargs

    # ── Stop sequences ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stop_sequences_forwarded(self):
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[Message.user("Hi")],
                    stop=["###", "END"],
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert kwargs["stop_sequences"] == ["###", "END"]

    # ── Model override ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_per_request_model_override(self):
        mock_resp = make_anthropic_message(model="claude-opus-4-6")
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client("claude-haiku-4-5-20251001")
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[Message.user("Hi")],
                    model="claude-opus-4-6",
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-6"

    # ── Tool use ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tool_use_response_parsed(self):
        mock_resp = make_anthropic_tool_message(
            tool_id="toolu_01XYZ",
            tool_name="get_weather",
            tool_input={"location": "London", "unit": "celsius"},
        )
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(LLMRequest(messages=[Message.user("Weather in London?")]))

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "toolu_01XYZ"
        assert tc.name == "get_weather"
        assert tc.arguments["location"] == "London"

    @pytest.mark.asyncio
    async def test_tool_definitions_forwarded(self):
        from conduit_sdk.models.requests import ToolDefinition

        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            tool = ToolDefinition(
                name="get_weather",
                description="Get current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            )
            await client.generate(
                LLMRequest(
                    messages=[Message.user("What's the weather?")],
                    tools=[tool],
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert "tools" in kwargs
        sent_tool = kwargs["tools"][0]
        assert sent_tool["name"] == "get_weather"
        assert "input_schema" in sent_tool  # Anthropic uses input_schema, not parameters

    # ── Tool finish reason ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_finish_reason_tool_calls(self):
        mock_resp = make_anthropic_tool_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(LLMRequest(messages=[Message.user("Hi")]))

        assert resp.finish_reason == FinishReason.TOOL_CALLS

    # ── Error wrapping ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_rate_limit_error_wrapped(self):
        import anthropic as ant

        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(
                side_effect=ant.RateLimitError(
                    message="rate limit",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                )
            )
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            with pytest.raises(RateLimitError):
                await client.generate(LLMRequest(messages=[Message.user("Hi")]))

    @pytest.mark.asyncio
    async def test_auth_error_wrapped(self):
        import anthropic as ant

        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(
                side_effect=ant.AuthenticationError(
                    message="invalid key",
                    response=MagicMock(status_code=401, headers={}),
                    body=None,
                )
            )
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            with pytest.raises(AuthenticationError):
                await client.generate(LLMRequest(messages=[Message.user("Hi")]))

    @pytest.mark.asyncio
    async def test_timeout_error_wrapped(self):
        import anthropic as ant

        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(
                side_effect=ant.APITimeoutError(request=MagicMock())
            )
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            with pytest.raises(TimeoutError):
                await client.generate(LLMRequest(messages=[Message.user("Hi")]))

    # ── Streaming ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stream_yields_text_chunks(self):
        stream_cm = make_anthropic_stream_context(["Hello", ", ", "world", "!"])
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.stream = MagicMock(return_value=stream_cm)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            chunks = []
            async for chunk in client.stream(LLMRequest(messages=[Message.user("Hi")])):
                chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world", "!"]

    @pytest.mark.asyncio
    async def test_stream_concatenated_equals_full_response(self):
        texts = ["The answer", " is ", "42."]
        stream_cm = make_anthropic_stream_context(texts)
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.stream = MagicMock(return_value=stream_cm)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            result = ""
            async for chunk in client.stream(LLMRequest(messages=[Message.user("Hi")])):
                result += chunk

        assert result == "The answer is 42."

    # ── Missing package ────────────────────────────────────────────────────

    def test_missing_anthropic_package_raises_import_error(self):
        import sys

        # Temporarily hide the anthropic package
        real_module = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            from conduit_sdk.providers.anthropic.llm import _require_anthropic

            with pytest.raises(ImportError, match="pip install llm-conduit\\[anthropic\\]"):
                _require_anthropic()
        finally:
            if real_module is not None:
                sys.modules["anthropic"] = real_module
            else:
                sys.modules.pop("anthropic", None)

    # ── Sync wrapper ───────────────────────────────────────────────────────

    def test_generate_sync(self):
        mock_resp = make_anthropic_message("Sync response works.")
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value
            resp = client.generate_sync(LLMRequest(messages=[Message.user("Hi")]))

        assert "Sync response works." in resp.content

    # ── Extra kwargs passthrough ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self):
        mock_resp = make_anthropic_message()
        with patch("anthropic.AsyncAnthropic") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.messages.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                LLMRequest(
                    messages=[Message.user("Hi")],
                    extra={"metadata": {"user_id": "u-123"}},
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert kwargs.get("metadata") == {"user_id": "u-123"}


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestAnthropicLLMClientIntegration:
    """
    Real API tests — require ANTHROPIC_API_KEY to be set.
    These are automatically skipped in CI unless the secret is present.
    Run locally:
        ANTHROPIC_API_KEY=sk-ant-... pytest -m integration tests/providers/test_anthropic_llm.py
    """

    def _client(self, model: str = "claude-haiku-4-5-20251001") -> AnthropicLLMClient:
        return AnthropicLLMClient(
            config=ClientConfig(
                provider="anthropic",
                model=model,
                cost=CostConfig(
                    input_cost_per_1k_tokens=0.00025,
                    output_cost_per_1k_tokens=0.00125,
                ),
            )
        )

    @pytest.mark.asyncio
    async def test_integration_basic_generate(self):
        client = self._client()
        resp = await client.generate(
            LLMRequest(
                messages=[Message.user("What is 2 + 2? Reply with just the number.")],
                max_tokens=10,
            )
        )

        assert resp.content.strip() != ""
        assert resp.provider == "anthropic"
        assert resp.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_integration_system_message(self):
        client = self._client()
        resp = await client.generate(
            LLMRequest(
                messages=[
                    Message.system("Always respond with exactly the word 'PONG' and nothing else."),
                    Message.user("PING"),
                ],
                max_tokens=10,
            )
        )

        assert "PONG" in resp.content.upper()

    @pytest.mark.asyncio
    async def test_integration_multi_turn(self):
        client = self._client()
        resp = await client.generate(
            LLMRequest(
                messages=[
                    Message.user("My name is Eren."),
                    Message.assistant("Nice to meet you, Eren!"),
                    Message.user("What is my name?"),
                ],
                max_tokens=30,
            )
        )

        assert "Eren" in resp.content

    @pytest.mark.asyncio
    async def test_integration_streaming(self):
        client = self._client()
        chunks: list[str] = []
        async for chunk in client.stream(
            LLMRequest(
                messages=[Message.user("Count from 1 to 5, one number per line.")],
                max_tokens=60,
            )
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert any(str(n) in full_text for n in range(1, 6))

    @pytest.mark.asyncio
    async def test_integration_cost_populated(self):
        from conduit_sdk.core.middleware import MiddlewarePipeline
        from conduit_sdk.utils.cost import CostMiddleware

        client = AnthropicLLMClient(
            config=ClientConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                cost=CostConfig(
                    input_cost_per_1k_tokens=0.00025,
                    output_cost_per_1k_tokens=0.00125,
                ),
            ),
            middleware=MiddlewarePipeline([CostMiddleware()]),
        )
        resp = await client.generate(
            LLMRequest(
                messages=[Message.user("Say hello.")],
                max_tokens=20,
            )
        )

        assert resp.cost is not None
        assert resp.cost.total_cost >= 0.0

    @pytest.mark.asyncio
    async def test_integration_tool_use(self):
        from conduit_sdk.models.requests import ToolDefinition

        client = self._client()
        weather_tool = ToolDefinition(
            name="get_current_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        )
        resp = await client.generate(
            LLMRequest(
                messages=[Message.user("What's the weather in Istanbul? Use the weather tool.")],
                tools=[weather_tool],
                max_tokens=200,
            )
        )

        # Claude should call the tool
        assert resp.finish_reason == FinishReason.TOOL_CALLS
        assert len(resp.tool_calls) >= 1
        tc = resp.tool_calls[0]
        assert tc.name == "get_current_weather"
        assert "city" in tc.arguments
