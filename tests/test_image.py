"""Tests for ImageGenClient."""

from __future__ import annotations

import pytest

from conduit_sdk.models.requests import ImageGenRequest, ImageSize


@pytest.mark.asyncio
async def test_generate_returns_image(image_client):
    req = ImageGenRequest(prompt="A sunset over the ocean")
    resp = await image_client.generate(req)
    assert len(resp.images) == 1
    assert resp.images[0].url == "https://example.com/image.png"


@pytest.mark.asyncio
async def test_generate_first_convenience(image_client):
    req = ImageGenRequest(prompt="A cat")
    resp = await image_client.generate(req)
    assert resp.first is not None
    assert resp.first.seed == 42


@pytest.mark.asyncio
async def test_generate_usage(image_client):
    req = ImageGenRequest(prompt="Test", num_images=3)
    resp = await image_client.generate(req)
    assert resp.usage.image_count == 3


@pytest.mark.asyncio
async def test_generate_with_size(image_client):
    req = ImageGenRequest(
        prompt="Portrait",
        size=ImageSize(width=512, height=768),
    )
    resp = await image_client.generate(req)
    assert resp.first is not None


def test_generate_sync(image_client):
    req = ImageGenRequest(prompt="A mountain")
    resp = image_client.generate_sync(req)
    assert resp.first is not None


@pytest.mark.asyncio
async def test_empty_images_returns_none_for_first(image_client):
    from conduit_sdk.models.responses import ImageGenResponse

    resp = ImageGenResponse(images=[])
    assert resp.first is None
