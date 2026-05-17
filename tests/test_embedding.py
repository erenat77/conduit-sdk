"""Tests for EmbeddingClient."""

from __future__ import annotations

import pytest

from conduit_sdk.models.requests import EmbeddingRequest


@pytest.mark.asyncio
async def test_embed_returns_vectors(embedding_client):
    req = EmbeddingRequest(inputs=["hello", "world"])
    resp = await embedding_client.embed(req)
    assert len(resp.embeddings) == 2
    assert resp.embeddings[0].index == 0
    assert resp.embeddings[1].index == 1


@pytest.mark.asyncio
async def test_embed_vectors_convenience(embedding_client):
    req = EmbeddingRequest(inputs=["foo"])
    resp = await embedding_client.embed(req)
    vectors = resp.vectors
    assert len(vectors) == 1
    assert all(v == 1.0 for v in vectors[0])


@pytest.mark.asyncio
async def test_embed_dimensions_property(embedding_client):
    req = EmbeddingRequest(inputs=["test"])
    resp = await embedding_client.embed(req)
    assert resp.dimensions == 4  # MockEmbeddingClient.DIMS


@pytest.mark.asyncio
async def test_embed_usage(embedding_client):
    req = EmbeddingRequest(inputs=["a", "b", "c"])
    resp = await embedding_client.embed(req)
    assert resp.usage.embedding_count == 3


def test_embed_sync(embedding_client):
    req = EmbeddingRequest(inputs=["sync"])
    resp = embedding_client.embed_sync(req)
    assert resp.dimensions == 4
