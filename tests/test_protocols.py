"""Tests for Protocol structural checking."""

from __future__ import annotations

from conduit_sdk.core.protocols import (
    EmbeddingProtocol,
    ImageGenProtocol,
    LLMProtocol,
    VideoGenProtocol,
)


def test_mock_llm_satisfies_protocol(llm_client):
    assert isinstance(llm_client, LLMProtocol)


def test_mock_image_satisfies_protocol(image_client):
    assert isinstance(image_client, ImageGenProtocol)


def test_mock_video_satisfies_protocol(video_client):
    assert isinstance(video_client, VideoGenProtocol)


def test_mock_embedding_satisfies_protocol(embedding_client):
    assert isinstance(embedding_client, EmbeddingProtocol)


def test_arbitrary_object_does_not_satisfy_llm_protocol():
    class NotAClient:
        pass

    assert not isinstance(NotAClient(), LLMProtocol)
