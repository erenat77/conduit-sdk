"""
OpenAIImageClient — adapter for OpenAI Images API (DALL-E 3 / DALL-E 2).
"""

from __future__ import annotations

import os
from typing import Any

from conduit_sdk.clients.image import ImageGenClient
from conduit_sdk.core.exceptions import ProviderError
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import ImageGenRequest
from conduit_sdk.models.responses import GeneratedImage, ImageGenResponse
from conduit_sdk.providers.openai.llm import _require_openai, _wrap_openai_error

# DALL-E 3 supports only specific size strings
_DALLE3_SIZES = {"1024x1024", "1792x1024", "1024x1792"}
_DALLE2_SIZES = {"256x256", "512x512", "1024x1024"}


class OpenAIImageClient(ImageGenClient):
    """
    Image generation client for OpenAI's Images API (DALL-E 3 and DALL-E 2).

    Supported models:
      - ``dall-e-3``  — highest quality, 1 image per call, prompt revision
      - ``dall-e-2``  — cheaper, up to 10 images per call

    Note: DALL-E 3 only accepts specific sizes: 1024×1024, 1792×1024, 1024×1792.
    The client will raise a ``ProviderError`` if an unsupported size is requested.

    Example::

        client = OpenAIImageClient(ClientConfig(
            model="dall-e-3",
            api_key="sk-...",
            cost=CostConfig(image_cost_per_unit=0.04),
        ))
        response = await client.generate(ImageGenRequest(
            prompt="A photorealistic sunset over the Sahara desert",
            size=ImageSize(width=1792, height=1024),
            output_format="png",
        ))
        print(response.first.url)
        print(response.first.revised_prompt)  # DALL-E 3 returns a rewritten prompt
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("OPENAI_API_KEY", "")

    def _sdk_client(self):
        openai = _require_openai()
        return openai.AsyncOpenAI(
            api_key=self._api_key(),
            base_url=self.config.api_base_url or None,
            timeout=self.config.timeout_seconds,
        )

    def _validate_size(self, request: ImageGenRequest, model: str) -> str:
        size_str = str(request.size)
        allowed = _DALLE3_SIZES if "dall-e-3" in model else _DALLE2_SIZES
        if size_str not in allowed:
            raise ProviderError(
                f"Model {model!r} does not support size {size_str!r}. "
                f"Allowed sizes: {sorted(allowed)}",
                provider="openai",
            )
        return size_str

    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        model = request.model or self.config.model
        size_str = self._validate_size(request, model)

        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "n": request.num_images,
            "size": size_str,
            "response_format": "url",
        }

        # DALL-E 3 specific options
        if "dall-e-3" in model:
            quality = request.extra.pop("quality", "standard")
            kwargs["quality"] = quality
            kwargs["style"] = request.extra.pop("style", "vivid")

        kwargs.update(request.extra)

        try:
            raw = await self._sdk_client().images.generate(**kwargs)
        except Exception as exc:
            raise _wrap_openai_error(exc) from exc

        images = [
            GeneratedImage(
                url=item.url,
                revised_prompt=item.revised_prompt,
                mime_type=f"image/{request.output_format}",
            )
            for item in raw.data
        ]

        return ImageGenResponse(
            images=images,
            usage=Usage(image_count=len(images)),
            model=model,
            provider="openai",
            raw_response=raw,
        )
