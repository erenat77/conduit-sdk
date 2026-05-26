"""
Unit tests for ReplicateImageClient and ReplicateVideoClient.

All tests mock the replicate SDK — no API key required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from conduit_sdk.core.config import ClientConfig
from conduit_sdk.models.requests import ImageGenRequest, VideoGenRequest
from conduit_sdk.providers.replicate import ReplicateImageClient, ReplicateVideoClient
from tests.providers.conftest import bare_pipeline


def _replicate_config(model: str = "black-forest-labs/flux-1.1-pro") -> ClientConfig:
    return ClientConfig(provider="replicate", model=model, api_key="r8_test")


def _image_client() -> ReplicateImageClient:
    return ReplicateImageClient(config=_replicate_config(), middleware=bare_pipeline())


def _video_client() -> ReplicateVideoClient:
    return ReplicateVideoClient(
        config=_replicate_config("minimax/video-01"), middleware=bare_pipeline()
    )


class TestReplicateImageClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_image_response(self):
        output_urls = ["https://replicate.delivery/abc/image.png"]
        client = _image_client()
        mock_sdk = MagicMock()
        mock_sdk.async_run = AsyncMock(return_value=output_urls)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(ImageGenRequest(prompt="A red fox in the snow"))

        assert len(resp.images) == 1
        assert resp.images[0].url == output_urls[0]
        assert resp.provider == "replicate"
        assert resp.model == "black-forest-labs/flux-1.1-pro"

    @pytest.mark.asyncio
    async def test_generate_passes_prompt(self):
        output_urls = ["https://replicate.delivery/abc/image.png"]
        client = _image_client()
        mock_sdk = MagicMock()
        run_mock = AsyncMock(return_value=output_urls)
        mock_sdk.async_run = run_mock
        client._sdk_client = lambda: mock_sdk

        await client.generate(ImageGenRequest(prompt="A mountain at dawn"))

        call_args = run_mock.call_args
        assert call_args.args[0] == "black-forest-labs/flux-1.1-pro"
        assert call_args.kwargs["input"]["prompt"] == "A mountain at dawn"

    @pytest.mark.asyncio
    async def test_generate_passes_size(self):
        output_urls = ["https://replicate.delivery/abc/image.png"]
        client = _image_client()
        mock_sdk = MagicMock()
        run_mock = AsyncMock(return_value=output_urls)
        mock_sdk.async_run = run_mock
        client._sdk_client = lambda: mock_sdk

        from conduit_sdk.models.requests import ImageSize

        req = ImageGenRequest(
            prompt="test",
            size=ImageSize(width=512, height=768),
        )
        await client.generate(req)

        input_data = run_mock.call_args.kwargs["input"]
        assert input_data["width"] == 512
        assert input_data["height"] == 768

    @pytest.mark.asyncio
    async def test_generate_handles_single_url_string(self):
        output = "https://replicate.delivery/abc/image.png"
        client = _image_client()
        mock_sdk = MagicMock()
        mock_sdk.async_run = AsyncMock(return_value=output)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(ImageGenRequest(prompt="test"))

        assert len(resp.images) == 1
        assert "replicate.delivery" in resp.images[0].url

    @pytest.mark.asyncio
    async def test_generate_multiple_images(self):
        output_urls = [
            "https://replicate.delivery/abc/img1.png",
            "https://replicate.delivery/abc/img2.png",
        ]
        client = _image_client()
        mock_sdk = MagicMock()
        mock_sdk.async_run = AsyncMock(return_value=output_urls)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(ImageGenRequest(prompt="test", num_images=2))

        assert len(resp.images) == 2
        assert resp.usage.image_count == 2

    def test_missing_replicate_package_raises(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "replicate":
                raise ImportError("No module named 'replicate'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from conduit_sdk.providers.replicate.image import _require_replicate

        with pytest.raises(ImportError, match="pip install llm-conduit\\[replicate\\]"):
            _require_replicate()


class TestReplicateVideoClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_video_response(self):
        output_urls = ["https://replicate.delivery/abc/video.mp4"]
        client = _video_client()
        mock_sdk = MagicMock()
        mock_sdk.async_run = AsyncMock(return_value=output_urls)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(VideoGenRequest(prompt="A drone shot of mountains"))

        assert len(resp.videos) == 1
        assert resp.videos[0].url == output_urls[0]
        assert resp.provider == "replicate"
        assert resp.model == "minimax/video-01"

    @pytest.mark.asyncio
    async def test_generate_passes_prompt_and_duration(self):
        output_urls = ["https://replicate.delivery/abc/video.mp4"]
        client = _video_client()
        mock_sdk = MagicMock()
        run_mock = AsyncMock(return_value=output_urls)
        mock_sdk.async_run = run_mock
        client._sdk_client = lambda: mock_sdk

        req = VideoGenRequest(prompt="Ocean waves", duration_seconds=5.0, fps=24)
        await client.generate(req)

        input_data = run_mock.call_args.kwargs["input"]
        assert input_data["prompt"] == "Ocean waves"
        assert input_data["duration"] == pytest.approx(5.0)
        assert input_data["fps"] == 24

    @pytest.mark.asyncio
    async def test_generate_handles_single_url(self):
        output = "https://replicate.delivery/abc/video.mp4"
        client = _video_client()
        mock_sdk = MagicMock()
        mock_sdk.async_run = AsyncMock(return_value=output)
        client._sdk_client = lambda: mock_sdk

        resp = await client.generate(VideoGenRequest(prompt="test"))

        assert len(resp.videos) == 1

    @pytest.mark.asyncio
    async def test_generate_passes_reference_image(self):
        output_urls = ["https://replicate.delivery/abc/video.mp4"]
        client = _video_client()
        mock_sdk = MagicMock()
        run_mock = AsyncMock(return_value=output_urls)
        mock_sdk.async_run = run_mock
        client._sdk_client = lambda: mock_sdk

        req = VideoGenRequest(
            prompt="Animate this image",
            reference_image_url="https://example.com/image.png",
        )
        await client.generate(req)

        input_data = run_mock.call_args.kwargs["input"]
        assert input_data["image"] == "https://example.com/image.png"
