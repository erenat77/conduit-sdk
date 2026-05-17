"""
Unit tests for MistralLLMClient and MistralEmbeddingClient.

All tests mock the mistralai SDK — no API key required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from conduit_sdk.core.config import ClientConfig
from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import EmbeddingRequest, LLMRequest
from conduit_sdk.providers.mistral import MistralEmbeddingClient, MistralLLMClient
from tests.providers.conftest import bare_pipeline


def _mistral_config(model: str = "mistral-large-latest") -> ClientConfig:
    return ClientConfig(provider="mistral", model=model, api_key="test_mistral_key")


def _llm_client() -> MistralLLMClient:
    return MistralLLMClient(config=_mistral_config(), middleware=bare_pipeline())


def _embed_client() -> MistralEmbeddingClient:
    return MistralEmbeddingClient(
        config=_mistral_config("mistral-embed"), middleware=bare_pipeline()
    )


def _make_chat_response(
    content: str = "Mistral mock answer",
    model: str = "mistral-large-latest",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = MagicMock()
    choice.finish_reason.__str__ = lambda self: f"FinishReason.{finish_reason}"
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


def _make_embed_response(vectors: list[list[float]], model: str = "mistral-embed") -> MagicMock:
    items = []
    for i, v in enumerate(vectors):
        item = MagicMock()
        item.index = i
        item.embedding = v
        items.append(item)
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.total_tokens = 5
    resp = MagicMock()
    resp.data = items
    resp.usage = usage
    resp.model = model
    return resp


class TestMistralLLMClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_response(self):
        mock_resp = _make_chat_response(content="Mixture of experts explained.")
        client = _llm_client()
        mock_sdk = MagicMock()
        mock_sdk.chat.complete_async = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(LLMRequest(messages=[Message.user("What is MoE?")]))

        assert resp.content == "Mixture of experts explained."
        assert resp.provider == "mistral"
        assert resp.model == "mistral-large-latest"

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self):
        mock_resp = _make_chat_response(prompt_tokens=12, completion_tokens=24)
        client = _llm_client()
        mock_sdk = MagicMock()
        mock_sdk.chat.complete_async = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(LLMRequest(messages=[Message.user("test")]))

        assert resp.usage.prompt_tokens == 12
        assert resp.usage.completion_tokens == 24
        assert resp.usage.total_tokens == 36

    @pytest.mark.asyncio
    async def test_generate_passes_params(self):
        mock_resp = _make_chat_response()
        client = _llm_client()
        mock_sdk = MagicMock()
        complete_mock = AsyncMock(return_value=mock_resp)
        mock_sdk.chat.complete_async = complete_mock
        client._sdk_client = lambda: mock_sdk

        req = LLMRequest(
            messages=[Message.user("test")],
            temperature=0.7,
            max_tokens=100,
        )
        await client.generate(req)

        kwargs = complete_mock.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        texts = ["Mis", "tral ", "streams!"]

        async def _async_gen():
            for t in texts:
                delta = MagicMock()
                delta.content = t
                choice = MagicMock()
                choice.delta = delta
                data = MagicMock()
                data.choices = [choice]
                event = MagicMock()
                event.data = data
                yield event

        client = _llm_client()
        mock_sdk = MagicMock()
        mock_sdk.chat.stream_async = AsyncMock(return_value=_async_gen())
        client._sdk_client = lambda: mock_sdk

        collected = []
        async for chunk in client.stream(LLMRequest(messages=[Message.user("Go!")])):
            collected.append(chunk)

        assert collected == texts

    def test_missing_mistral_package_raises(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mistralai":
                raise ImportError("No module named 'mistralai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from conduit_sdk.providers.mistral.llm import _require_mistral

        with pytest.raises(ImportError, match="pip install llm-conduit\\[mistral\\]"):
            _require_mistral()


class TestMistralEmbeddingClientUnit:
    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self):
        vectors = [[0.1] * 1024, [0.2] * 1024]
        mock_resp = _make_embed_response(vectors)
        client = _embed_client()
        mock_sdk = MagicMock()
        mock_sdk.embeddings.create_async = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk

        resp = await client.embed(
            EmbeddingRequest(inputs=["text one", "text two"])
        )

        assert len(resp.embeddings) == 2
        assert len(resp.vectors[0]) == 1024
        assert resp.provider == "mistral"
        assert resp.dimensions == 1024

    @pytest.mark.asyncio
    async def test_embed_passes_model(self):
        vectors = [[0.1] * 1024]
        mock_resp = _make_embed_response(vectors)
        client = _embed_client()
        mock_sdk = MagicMock()
        create_mock = AsyncMock(return_value=mock_resp)
        mock_sdk.embeddings.create_async = create_mock
        client._sdk_client = lambda: mock_sdk

        await client.embed(EmbeddingRequest(inputs=["test"]))

        assert create_mock.call_args.kwargs["model"] == "mistral-embed"
