"""
Unit tests for GeminiLLMClient and GeminiEmbeddingClient.

All tests mock the google-genai SDK — no API key required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from conduit_sdk.core.config import ClientConfig
from conduit_sdk.models.common import Message
from conduit_sdk.models.requests import EmbeddingRequest, LLMRequest
from conduit_sdk.providers.gemini import GeminiEmbeddingClient, GeminiLLMClient
from tests.providers.conftest import bare_pipeline


def _gemini_config(model: str = "gemini-2.0-flash") -> ClientConfig:
    return ClientConfig(provider="gemini", model=model, api_key="AIza_test")


def _llm_client() -> GeminiLLMClient:
    return GeminiLLMClient(config=_gemini_config(), middleware=bare_pipeline())


def _embed_client() -> GeminiEmbeddingClient:
    return GeminiEmbeddingClient(
        config=_gemini_config("text-embedding-004"), middleware=bare_pipeline()
    )


def _make_generate_response(
    text: str = "Gemini mock answer",
    finish_reason: str = "STOP",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    usage.total_token_count = prompt_tokens + completion_tokens

    candidate = MagicMock()
    candidate.finish_reason = MagicMock()
    candidate.finish_reason.__str__ = lambda self: f"FinishReason.{finish_reason}"

    resp = MagicMock()
    resp.text = text
    resp.candidates = [candidate]
    resp.usage_metadata = usage
    return resp


def _make_embed_response(vectors: list[list[float]]) -> MagicMock:
    embeddings = []
    for v in vectors:
        e = MagicMock()
        e.values = v
        embeddings.append(e)
    resp = MagicMock()
    resp.embeddings = embeddings
    return resp


class TestGeminiLLMClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_response(self):
        mock_resp = _make_generate_response(text="Attention is all you need.")
        client = _llm_client()
        mock_sdk_client = MagicMock()
        mock_sdk_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk_client

        resp = await client.generate(LLMRequest(messages=[Message.user("Explain attention.")]))

        assert resp.content == "Attention is all you need."
        assert resp.provider == "gemini"
        assert resp.model == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self):
        mock_resp = _make_generate_response(prompt_tokens=15, completion_tokens=35)
        client = _llm_client()
        mock_sdk_client = MagicMock()
        mock_sdk_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk_client

        resp = await client.generate(LLMRequest(messages=[Message.user("test")]))

        assert resp.usage.prompt_tokens == 15
        assert resp.usage.completion_tokens == 35
        assert resp.usage.total_tokens == 50

    @pytest.mark.asyncio
    async def test_system_message_extracted(self):
        mock_resp = _make_generate_response()
        client = _llm_client()
        mock_sdk_client = MagicMock()
        create_mock = AsyncMock(return_value=mock_resp)
        mock_sdk_client.aio.models.generate_content = create_mock
        client._sdk_client = lambda: mock_sdk_client

        req = LLMRequest(
            messages=[
                Message.system("You are an expert."),
                Message.user("Explain RL."),
            ]
        )
        await client.generate(req)

        call_kwargs = create_mock.call_args.kwargs
        # system instruction should be in config, not in contents
        assert call_kwargs["config"] is not None
        contents = call_kwargs["contents"]
        # contents should only have the user message
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_assistant_role_mapped_to_model(self):
        mock_resp = _make_generate_response()
        client = _llm_client()
        mock_sdk_client = MagicMock()
        create_mock = AsyncMock(return_value=mock_resp)
        mock_sdk_client.aio.models.generate_content = create_mock
        client._sdk_client = lambda: mock_sdk_client

        req = LLMRequest(
            messages=[
                Message.user("Hello"),
                Message.assistant("Hi there!"),
                Message.user("How are you?"),
            ]
        )
        await client.generate(req)

        contents = create_mock.call_args.kwargs["contents"]
        roles = [c["role"] for c in contents]
        assert roles == ["user", "model", "user"]

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        texts = ["Gem", "ini ", "rocks!"]

        async def _async_gen():
            for t in texts:
                chunk = MagicMock()
                chunk.text = t
                yield chunk

        client = _llm_client()
        mock_sdk_client = MagicMock()
        mock_sdk_client.aio.models.generate_content_stream = AsyncMock(return_value=_async_gen())
        client._sdk_client = lambda: mock_sdk_client

        collected = []
        async for chunk in client.stream(LLMRequest(messages=[Message.user("Go!")])):
            collected.append(chunk)

        assert collected == texts

    def test_missing_genai_raises(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "google.genai":
                raise ImportError("No module named 'google.genai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from conduit_sdk.providers.gemini.llm import _require_genai

        with pytest.raises(ImportError, match="pip install llm-conduit\\[gemini\\]"):
            _require_genai()


class TestGeminiEmbeddingClientUnit:
    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self):
        vectors = [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256]
        mock_resp = _make_embed_response(vectors)
        client = _embed_client()
        mock_sdk_client = MagicMock()
        mock_sdk_client.aio.models.embed_content = AsyncMock(return_value=mock_resp)
        client._sdk_client = lambda: mock_sdk_client

        resp = await client.embed(EmbeddingRequest(inputs=["Hello world", "Goodbye world"]))

        assert len(resp.embeddings) == 2
        assert len(resp.vectors[0]) == 768
        assert resp.provider == "gemini"
        assert resp.model == "text-embedding-004"

    @pytest.mark.asyncio
    async def test_embed_passes_task_type(self):
        vectors = [[0.1] * 768]
        mock_resp = _make_embed_response(vectors)
        client = _embed_client()
        mock_sdk_client = MagicMock()
        create_mock = AsyncMock(return_value=mock_resp)
        mock_sdk_client.aio.models.embed_content = create_mock
        client._sdk_client = lambda: mock_sdk_client

        await client.embed(EmbeddingRequest(inputs=["test query"], input_type="query"))

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["config"]["task_type"] == "RETRIEVAL_QUERY"
