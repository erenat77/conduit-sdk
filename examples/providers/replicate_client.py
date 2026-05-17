"""
Provider skeleton — Replicate
==============================
Covers: ImageGenClient (any Replicate image model).

Install: pip install replicate
"""

from __future__ import annotations

from conduit_sdk.clients import ImageGenClient
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import ImageGenRequest
from conduit_sdk.models.responses import GeneratedImage, ImageGenResponse


class ReplicateImageClient(ImageGenClient):
    """
    Image-generation adapter for Replicate.

    The ``config.model`` field should be the full Replicate model string,
    e.g. ``"stability-ai/stable-diffusion-3"``.

    Usage::

        client = ReplicateImageClient(ClientConfig(
            provider="replicate",
            model="stability-ai/stable-diffusion-3",
            api_key="r8_...",   # set REPLICATE_API_TOKEN env var instead
        ))
        response = await client.generate(ImageGenRequest(
            prompt="A photorealistic sunset over mountains",
            size=ImageSize(width=1024, height=768),
            steps=28,
        ))
        print(response.first.url)
    """

    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        import replicate  # noqa: PLC0415

        input_payload: dict = {
            "prompt": request.prompt,
            "width": request.size.width,
            "height": request.size.height,
            "num_outputs": request.num_images,
            "num_inference_steps": request.steps or 28,
            "guidance_scale": request.guidance_scale or 7.5,
        }
        if request.negative_prompt:
            input_payload["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            input_payload["seed"] = request.seed

        # Merge provider-specific extras
        input_payload.update(request.extra)

        output = await replicate.async_run(
            self.config.model,
            input=input_payload,
        )

        # Replicate returns a list of URLs (or FileOutput objects)
        urls = [str(url) for url in output]

        return ImageGenResponse(
            images=[GeneratedImage(url=url) for url in urls],
            usage=Usage(image_count=len(urls)),
            model=self.config.model,
            provider="replicate",
        )
