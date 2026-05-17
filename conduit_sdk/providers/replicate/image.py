"""
ReplicateImageClient — adapter for Replicate image generation models.

Uses the official ``replicate`` Python SDK.

Supported models (pass as config.model or request.model):
  - "black-forest-labs/flux-1.1-pro"
  - "black-forest-labs/flux-schnell"
  - "stability-ai/sdxl:..."
  - any Replicate image model version string

Install: pip install llm-conduit[replicate]
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from conduit_sdk.clients.image import ImageGenClient
from conduit_sdk.core.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import ImageGenRequest
from conduit_sdk.models.responses import GeneratedImage, ImageGenResponse

if TYPE_CHECKING:
    pass


def _require_replicate():
    try:
        import replicate  # noqa: PLC0415

        return replicate
    except ImportError:
        raise ImportError(
            "Replicate provider requires the replicate package. "
            "Install it with: pip install llm-conduit[replicate]"
        ) from None


def _wrap_replicate_error(exc: Exception, provider: str = "replicate") -> ProviderError:
    exc_type = type(exc).__name__
    exc_str = str(exc)
    if "Unauthorized" in exc_type or "authentication" in exc_str.lower() or "401" in exc_str:
        return AuthenticationError(exc_str, provider=provider)
    if "RateLimit" in exc_type or "429" in exc_str or "rate limit" in exc_str.lower():
        return RateLimitError(exc_str, provider=provider)
    if "Timeout" in exc_type or "timeout" in exc_str.lower():
        return TimeoutError(exc_str, provider=provider)
    return ProviderError(exc_str, provider=provider)


class ReplicateImageClient(ImageGenClient):
    """
    Image generation client adapter for Replicate.

    Replicate hosts hundreds of open-source image models. Pass the model
    as ``config.model`` or ``request.model`` in the format
    ``"owner/model-name"`` or ``"owner/model-name:version-sha"``.

    The API token is resolved in this order:
      1. ``config.api_key``
      2. ``REPLICATE_API_TOKEN`` environment variable

    Example::

        from conduit_sdk.providers.replicate import ReplicateImageClient
        from conduit_sdk.core.config import ClientConfig

        client = ReplicateImageClient(ClientConfig(
            provider="replicate",
            model="black-forest-labs/flux-1.1-pro",
            api_key="r8_...",   # or set REPLICATE_API_TOKEN env var
        ))
        response = await client.generate(ImageGenRequest(
            prompt="A photorealistic red fox in a snowy forest",
        ))
        print(response.first.url)
    """

    def _api_token(self) -> str:
        return self.config.api_key or os.environ.get("REPLICATE_API_TOKEN", "")

    def _sdk_client(self):
        replicate = _require_replicate()
        return replicate.Client(api_token=self._api_token())

    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        model = request.model or self.config.model

        # Build the input dict for Replicate's model
        input_data: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.size.width if request.size else 1024,
            "height": request.size.height if request.size else 1024,
            "num_outputs": request.num_images,
            "output_format": request.output_format,
        }
        input_data.update(request.extra)

        try:
            client = self._sdk_client()
            output = await client.async_run(model, input=input_data)
        except Exception as exc:
            raise _wrap_replicate_error(exc) from exc

        # Replicate returns a list of URLs (FileOutput objects or strings)
        urls: list[str] = []
        if isinstance(output, list):
            for item in output:
                url = str(item)  # FileOutput.__str__() returns the URL
                urls.append(url)
        elif output:
            urls.append(str(output))

        images = [
            GeneratedImage(
                url=url,
                mime_type=f"image/{request.output_format}",
            )
            for url in urls
        ]

        return ImageGenResponse(
            images=images,
            usage=Usage(image_count=len(images)),
            model=model,
            provider="replicate",
            raw_response=output,
        )
