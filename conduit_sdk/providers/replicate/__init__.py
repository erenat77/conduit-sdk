"""
Replicate provider — pip install llm-conduit[replicate]

Open-source model hosting via Replicate's cloud inference API.

    ReplicateImageClient → ImageGenClient  (flux-1.1-pro, sdxl, …)
    ReplicateVideoClient → VideoGenClient  (minimax/video-01, luma, …)

Quick start::

    from conduit_sdk.providers.replicate import ReplicateImageClient
    from conduit_sdk.core.config import ClientConfig
    from conduit_sdk.models.requests import ImageGenRequest

    client = ReplicateImageClient(ClientConfig(
        provider="replicate",
        model="black-forest-labs/flux-1.1-pro",
        api_key="r8_...",   # or set REPLICATE_API_TOKEN env var
    ))
    response = await client.generate(ImageGenRequest(
        prompt="A majestic lion in the savannah at sunset",
    ))
    print(response.first.url)
"""

from conduit_sdk.providers.replicate.image import ReplicateImageClient
from conduit_sdk.providers.replicate.video import ReplicateVideoClient

__all__ = ["ReplicateImageClient", "ReplicateVideoClient"]
