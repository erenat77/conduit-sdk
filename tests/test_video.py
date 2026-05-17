"""Tests for VideoGenClient."""

from __future__ import annotations

import pytest

from conduit_sdk.models.requests import VideoGenRequest


@pytest.mark.asyncio
async def test_generate_returns_video(video_client):
    req = VideoGenRequest(prompt="A timelapse of clouds")
    resp = await video_client.generate(req)
    assert len(resp.videos) == 1
    assert resp.first is not None
    assert resp.first.url == "https://example.com/video.mp4"


@pytest.mark.asyncio
async def test_generate_duration_preserved(video_client):
    req = VideoGenRequest(prompt="Ocean waves", duration_seconds=8.0, fps=30)
    resp = await video_client.generate(req)
    assert resp.first.duration_seconds == 8.0
    assert resp.first.fps == 30


@pytest.mark.asyncio
async def test_generate_usage_records_seconds(video_client):
    req = VideoGenRequest(prompt="Fireworks", duration_seconds=5.0)
    resp = await video_client.generate(req)
    assert resp.usage.video_seconds == 5.0


def test_generate_sync(video_client):
    req = VideoGenRequest(prompt="Stars")
    resp = video_client.generate_sync(req)
    assert resp.first is not None
