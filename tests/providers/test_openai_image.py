"""
Tests for OpenAIImageClient (DALL-E 3 / DALL-E 2).

Unit tests   — mocked SDK, always run.
Integration  — real API, requires OPENAI_API_KEY.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.core.exceptions import ProviderError
from conduit_sdk.models.requests import ImageGenRequest, ImageSize
from conduit_sdk.providers.openai import OpenAIImageClient
from tests.providers.conftest import (
    bare_pipeline,
    make_image_response,
    openai_config,
)


def _client(model: str = "dall-e-3") -> OpenAIImageClient:
    return OpenAIImageClient(config=openai_config(model), middleware=bare_pipeline())


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAIImageClientUnit:
    @pytest.mark.asyncio
    async def test_generate_returns_image_url(self):
        mock_resp = make_image_response(
            ["https://oaidalleapiprodscus.blob.core.windows.net/img.png"]
        )
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.images.generate = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(ImageGenRequest(prompt="A sunset"))

        assert resp.first is not None
        assert "https://" in resp.first.url

    @pytest.mark.asyncio
    async def test_generate_returns_revised_prompt(self):
        mock_resp = make_image_response(
            ["https://example.com/img.png"],
            revised_prompt="A stunning photorealistic sunset over mountains",
        )
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.images.generate = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(ImageGenRequest(prompt="A sunset"))

        assert resp.first.revised_prompt is not None

    @pytest.mark.asyncio
    async def test_generate_uses_correct_size_format(self):
        mock_resp = make_image_response(["https://example.com/img.png"])
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.images.generate = create_mock
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                ImageGenRequest(
                    prompt="A sunset",
                    size=ImageSize(width=1792, height=1024),
                )
            )

        assert create_mock.call_args.kwargs["size"] == "1792x1024"

    @pytest.mark.asyncio
    async def test_generate_invalid_size_raises_provider_error(self):
        client = _client("dall-e-3")
        with pytest.raises(ProviderError, match="does not support size"):
            await client.generate(
                ImageGenRequest(
                    prompt="A sunset",
                    size=ImageSize(width=300, height=300),  # not a valid DALL-E 3 size
                )
            )

    @pytest.mark.asyncio
    async def test_generate_dalle2_valid_size(self):
        mock_resp = make_image_response(["https://example.com/img.png"])
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.images.generate = create_mock
            client = _client("dall-e-2")
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                ImageGenRequest(
                    prompt="A cat",
                    size=ImageSize(width=512, height=512),
                )
            )

        assert create_mock.call_args.kwargs["size"] == "512x512"

    @pytest.mark.asyncio
    async def test_generate_dalle3_passes_quality_and_style(self):
        mock_resp = make_image_response(["https://example.com/img.png"])
        with patch("openai.AsyncOpenAI") as MockSDK:
            create_mock = AsyncMock(return_value=mock_resp)
            MockSDK.return_value.images.generate = create_mock
            client = _client("dall-e-3")
            client._sdk_client = lambda: MockSDK.return_value

            await client.generate(
                ImageGenRequest(
                    prompt="A sunset",
                    extra={"quality": "hd", "style": "natural"},
                )
            )

        kwargs = create_mock.call_args.kwargs
        assert kwargs["quality"] == "hd"
        assert kwargs["style"] == "natural"

    @pytest.mark.asyncio
    async def test_generate_usage_image_count(self):
        mock_resp = make_image_response(["https://example.com/img.png"])
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.images.generate = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(ImageGenRequest(prompt="test"))

        assert resp.usage.image_count == 1

    @pytest.mark.asyncio
    async def test_generate_provider_is_openai(self):
        mock_resp = make_image_response(["https://example.com/img.png"])
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.images.generate = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value

            resp = await client.generate(ImageGenRequest(prompt="test"))

        assert resp.provider == "openai"

    def test_generate_sync(self):
        mock_resp = make_image_response(["https://example.com/sync.png"])
        with patch("openai.AsyncOpenAI") as MockSDK:
            MockSDK.return_value.images.generate = AsyncMock(return_value=mock_resp)
            client = _client()
            client._sdk_client = lambda: MockSDK.return_value
            resp = client.generate_sync(ImageGenRequest(prompt="sync test"))

        assert resp.first.url == "https://example.com/sync.png"


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestOpenAIImageClientIntegration:
    def _client(self) -> OpenAIImageClient:
        return OpenAIImageClient(
            config=ClientConfig(
                provider="openai",
                model="dall-e-3",
                cost=CostConfig(image_cost_per_unit=0.04),
            )
        )

    @pytest.mark.asyncio
    async def test_integration_generate_returns_url(self):
        client = self._client()
        resp = await client.generate(
            ImageGenRequest(
                prompt="A simple red circle on a white background",
                size=ImageSize(width=1024, height=1024),
            )
        )

        assert resp.first is not None
        assert resp.first.url.startswith("https://")
        assert resp.first.revised_prompt is not None
        assert resp.usage.image_count == 1
        assert resp.provider == "openai"

    @pytest.mark.asyncio
    async def test_integration_hd_quality(self):
        client = self._client()
        resp = await client.generate(
            ImageGenRequest(
                prompt="A photorealistic mountain landscape at dawn",
                size=ImageSize(width=1792, height=1024),
                extra={"quality": "hd"},
            )
        )

        assert resp.first.url.startswith("https://")
