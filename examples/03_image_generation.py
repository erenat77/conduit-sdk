"""
Example 03 — Image generation
==============================
Shows:
  - Implementing ImageGenClient
  - Using ImageSize, negative prompts, seeds for reproducibility
  - Accessing GeneratedImage fields (url, seed, mime_type)
  - num_images > 1 and iterating results
"""

from __future__ import annotations

import asyncio

from conduit_sdk.clients import ImageGenClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import ImageGenRequest, ImageSize
from conduit_sdk.models.responses import GeneratedImage, ImageGenResponse

# ---------------------------------------------------------------------------
# Mock image provider
# ---------------------------------------------------------------------------


class MockImageClient(ImageGenClient):
    """
    Simulates an image generation provider.
    Replace _generate with a real call (Stability AI, DALL-E, Midjourney, etc.)
    """

    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        images = [
            GeneratedImage(
                url=(
                    f"https://mock-cdn.example.com/images/"
                    f"gen_{i}_{request.seed or 0}.{request.output_format}"
                ),
                seed=(request.seed or 42) + i,
                revised_prompt=f"[Revised] {request.prompt}",
                mime_type=f"image/{request.output_format}",
            )
            for i in range(request.num_images)
        ]
        return ImageGenResponse(
            images=images,
            usage=Usage(image_count=request.num_images),
            model=self.config.model,
            provider=self.config.provider,
        )


client = MockImageClient(
    config=ClientConfig(provider="mock", model="mock-diffusion-v3"),
    middleware=MiddlewarePipeline([]),
)


async def main() -> None:
    print("=" * 55)
    print("Example 03 — Image Generation")
    print("=" * 55)

    # --- Single image ---
    print("\n[Single image — portrait]")
    response = await client.generate(
        ImageGenRequest(
            prompt="A serene Japanese garden at dawn, cherry blossoms, soft mist",
            negative_prompt="people, crowds, noise, ugly, blurry",
            size=ImageSize(width=768, height=1024),
            steps=30,
            guidance_scale=7.5,
            seed=12345,
            output_format="png",
        )
    )

    img = response.first
    print(f"  URL          : {img.url}")
    print(f"  Seed         : {img.seed}")
    print(f"  Revised prompt: {img.revised_prompt}")
    print(f"  Format       : {img.mime_type}")

    # --- Multiple images in one call ---
    print("\n[4 images — landscape batch]")
    batch = await client.generate(
        ImageGenRequest(
            prompt="Futuristic city skyline at sunset, cyberpunk aesthetic",
            size=ImageSize(width=1024, height=576),  # 16:9 widescreen
            num_images=4,
            seed=99,
        )
    )

    for i, image in enumerate(batch.images):
        print(f"  Image {i + 1}: {image.url}  (seed={image.seed})")

    print(f"\nTotal images generated: {batch.usage.image_count}")


if __name__ == "__main__":
    asyncio.run(main())

    # Sync wrapper — must be called outside an event loop
    print("\n[Sync call]")
    sync_resp = client.generate_sync(ImageGenRequest(prompt="Abstract digital art, vibrant colors"))
    print(f"  Sync result: {sync_resp.first.url}")
