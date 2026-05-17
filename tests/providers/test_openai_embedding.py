"""
Tests for OpenAIEmbeddingClient.

Unit tests   — mocked SDK, always run.
Integration  — real API, requires OPENAI_API_KEY.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from conduit_sdk.core.config import ClientConfig
from conduit_sdk.models.requests import EmbeddingRequest
from conduit_sdk.providers.openai import OpenAIEmbeddingClient
from tests.providers.conftest import (
    bare_pipeline,
    make_embedding_response,
    openai_config,
)


def _client(model: str = "text-embedding-3-large") -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(config=openai_config(model), middleware=bare_pipeline())


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAIEmbeddingClientUnit:
    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self):
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_resp = make_embedding_response(vectors)
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.embeddings.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.embed(EmbeddingRequest(inputs=["hello", "world"]))

        assert len(resp.embeddings) == 2
        assert resp.vectors[0] == [0.1, 0.2, 0.3]
        assert resp.vectors[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_sets_provider_and_model(self):
        mock_resp = make_embedding_response([[0.1]])
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.embeddings.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.embed(EmbeddingRequest(inputs=["test"]))

        assert resp.provider == "openai"

    @pytest.mark.asyncio
    async def test_embed_passes_dimensions_for_mrl(self):
        mock_resp = make_embedding_response([[0.1] * 512])
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.embeddings.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.embed(EmbeddingRequest(inputs=["test"], dimensions=512))

        assert create_mock.call_args.kwargs["dimensions"] == 512

    @pytest.mark.asyncio
    async def test_embed_passes_encoding_format(self):
        mock_resp = make_embedding_response([[0.1]])
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.embeddings.create = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.embed(EmbeddingRequest(inputs=["test"], encoding_format="float"))

        assert create_mock.call_args.kwargs["encoding_format"] == "float"

    @pytest.mark.asyncio
    async def test_embed_usage_populated(self):
        mock_resp = make_embedding_response([[0.1, 0.2]], prompt_tokens=8)
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.embeddings.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.embed(EmbeddingRequest(inputs=["hello world"]))

        assert resp.usage.prompt_tokens == 8
        assert resp.usage.embedding_count == 1

    @pytest.mark.asyncio
    async def test_embed_index_order_preserved(self):
        vectors = [[float(i)] for i in range(5)]
        mock_resp = make_embedding_response(vectors)
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.embeddings.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.embed(EmbeddingRequest(inputs=[f"text {i}" for i in range(5)]))

        for i, emb in enumerate(resp.embeddings):
            assert emb.index == i

    def test_embed_sync(self):
        mock_resp = make_embedding_response([[1.0, 2.0]])
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.embeddings.create = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value
            resp = client.embed_sync(EmbeddingRequest(inputs=["sync test"]))

        assert resp.vectors[0] == [1.0, 2.0]


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestOpenAIEmbeddingClientIntegration:
    def _client(self, model: str = "text-embedding-3-small") -> OpenAIEmbeddingClient:
        return OpenAIEmbeddingClient(config=ClientConfig(provider="openai", model=model))

    @pytest.mark.asyncio
    async def test_integration_single_input(self):
        client = self._client()
        resp = await client.embed(EmbeddingRequest(inputs=["The quick brown fox"]))

        assert len(resp.embeddings) == 1
        assert resp.dimensions == 1536  # text-embedding-3-small default
        assert resp.usage.total_tokens > 0
        assert all(isinstance(v, float) for v in resp.vectors[0])

    @pytest.mark.asyncio
    async def test_integration_batch(self):
        client = self._client()
        inputs = ["Machine learning", "Deep learning", "Reinforcement learning"]
        resp = await client.embed(EmbeddingRequest(inputs=inputs))

        assert len(resp.embeddings) == 3
        assert resp.usage.embedding_count == 3
        # All vectors same dimensionality
        assert len({len(v) for v in resp.vectors}) == 1

    @pytest.mark.asyncio
    async def test_integration_mrl_dimensions(self):
        """MRL: reduce dimensions without re-embedding."""
        client = self._client("text-embedding-3-large")
        resp = await client.embed(EmbeddingRequest(inputs=["test"], dimensions=256))

        assert resp.dimensions == 256  # reduced from 3072

    @pytest.mark.asyncio
    async def test_integration_semantic_similarity(self):
        """Similar sentences should have higher cosine similarity than dissimilar ones."""
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(y**2 for y in b)))

        client = self._client()
        resp = await client.embed(
            EmbeddingRequest(
                inputs=[
                    "The cat sat on the mat",
                    "A feline rested on a rug",  # semantically similar
                    "Quantum physics is complex",  # semantically dissimilar
                ]
            )
        )

        v0, v1, v2 = resp.vectors
        sim_similar = cosine(v0, v1)
        sim_dissimilar = cosine(v0, v2)

        assert sim_similar > sim_dissimilar, (
            f"Expected similar sentences to score higher: {sim_similar:.4f} vs {sim_dissimilar:.4f}"
        )
