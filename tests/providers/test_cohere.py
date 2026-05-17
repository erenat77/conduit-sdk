"""
Unit tests for CohereLLMClient and CohereEmbeddingClient.

All tests mock the cohere SDK — no API key required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from conduit_sdk.core.config import ClientConfig
from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import EmbeddingRequest, LLMRequest
from conduit_sdk.providers.cohere import CohereEmbeddingClient, CohereLLMClient
from tests.providers.conftest import bare_pipeline


def _cohere_config(model: str = "command-r-plus-08-2024") -> ClientConfig:
    return ClientConfig(provider="cohere", model=model, api_key="test_cohere_key")


def _llm_client() -> CohereLLMClient:
    return CohereLLMClient(config=_cohere_config(), middleware=bare_pipeline())


def _embed_client() -> CohereEmbeddingClient:
    return CohereEmbeddingClient(
        config=_cohere_config("embed-english-v3.0"), middleware=bare_pipeline()
    )


def _make_content_block(text: str = "Cohere mock response") -> MagicMock:
    block = MagicMock()
    block.text = text
    return block


def _make_chat_response(
    content: str = "Cohere mock response",
    finish_reason: str = "COMPLETE",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> MagicMock:
    msg = MagicMock()
    msg.content = [_make_content_block(content)]
    billed = MagicMock()
    billed.input_tokens = input_tokens
    billed.output_tokens = output_tokens
    usage = MagicMock()
    usage.billed_units = billed
    resp = MagicMock()
    resp.message = msg
    resp.finish_reason = MagicMock()
    resp.finish_reason.__str__ = lambda self: f"FinishReason.{finish_reason}"
    resp.usage = usage
    return resp


def _make_embed_response(vectors: list[list[float]]) -> MagicMock:
    embeddings = MagicMock()
    embeddings.float_ = vectors
    resp = MagicMock()
    resp.embeddings = embeddings
    return resp


class TestCohereLLMClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_response(self):
        mock_resp = _make_chat_response(content="RAG combines retrieval and generation.")
        client = _llm_client()
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(LLMRequest(messages=[Message.user("What is RAG?")]))

        assert resp.content == "RAG combines retrieval and generation."
        assert resp.provider == "cohere"
        assert resp.model == "command-r-plus-08-2024"

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self):
        mock_resp = _make_chat_response(input_tokens=8, output_tokens=16)
        client = _llm_client()
        mock_sdk = MagicMock()
        mock_sdk.chat = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(LLMRequest(messages=[Message.user("test")]))

        assert resp.usage.prompt_tokens == 8
        assert resp.usage.completion_tokens == 16
        assert resp.usage.total_tokens == 24

    @pytest.mark.asyncio
    async def test_generate_passes_params(self):
        mock_resp = _make_chat_response()
        client = _llm_client()
        mock_sdk = MagicMock()
        chat_mock = AsyncMock(return_value=mock_resp)
        mock_sdk.chat = chat_mock
        client._sdk_client = lambda: mock_sdk

        req = LLMRequest(
            messages=[Message.user("test")],
            temperature=0.4,
            max_tokens=200,
            top_p=0.9,
        )
        await client.generate(req)

        kwargs = chat_mock.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.4)
        assert kwargs["max_tokens"] == 200
        assert kwargs["p"] == pytest.approx(0.9)  # Cohere uses 'p'

    @pytest.mark.asyncio
    async def test_system_message_sent(self):
        mock_resp = _make_chat_response()
        client = _llm_client()
        mock_sdk = MagicMock()
        chat_mock = AsyncMock(return_value=mock_resp)
        mock_sdk.chat = chat_mock
        client._sdk_client = lambda: mock_sdk

        req = LLMRequest(
            messages=[
                Message.system("Be concise."),
                Message.user("What is ML?"),
            ]
        )
        await client.generate(req)

        messages = chat_mock.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_missing_cohere_package_raises(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cohere":
                raise ImportError("No module named 'cohere'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from conduit_sdk.providers.cohere.llm import _require_cohere

        with pytest.raises(ImportError, match="pip install llm-conduit\\[cohere\\]"):
            _require_cohere()


class TestCohereEmbeddingClientUnit:
    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self):
        vectors = [[0.1] * 1024, [0.2] * 1024]
        mock_resp = _make_embed_response(vectors)
        client = _embed_client()
        mock_sdk = MagicMock()
        mock_sdk.embed = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk

        resp = await client.embed(
            EmbeddingRequest(inputs=["text one", "text two"], input_type="document")
        )

        assert len(resp.embeddings) == 2
        assert len(resp.vectors[0]) == 1024
        assert resp.provider == "cohere"
        assert resp.dimensions == 1024

    @pytest.mark.asyncio
    async def test_embed_maps_input_type(self):
        vectors = [[0.1] * 1024]
        mock_resp = _make_embed_response(vectors)
        client = _embed_client()
        mock_sdk = MagicMock()
        embed_mock = AsyncMock(return_value=mock_resp)
        mock_sdk.embed = embed_mock
        client._sdk_client = lambda: mock_sdk

        await client.embed(EmbeddingRequest(inputs=["query text"], input_type="query"))

        kwargs = embed_mock.call_args.kwargs
        assert kwargs["input_type"] == "search_query"

    @pytest.mark.asyncio
    async def test_embed_passes_dimensions(self):
        vectors = [[0.1] * 512]
        mock_resp = _make_embed_response(vectors)
        client = _embed_client()
        mock_sdk = MagicMock()
        embed_mock = AsyncMock(return_value=mock_resp)
        mock_sdk.embed = embed_mock
        client._sdk_client = lambda: mock_sdk

        await client.embed(EmbeddingRequest(inputs=["test"], dimensions=512))

        kwargs = embed_mock.call_args.kwargs
        assert kwargs["output_dimension"] == 512
