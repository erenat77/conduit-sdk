"""
ImageGenClient — abstract base for image-generation clients.

HOW TO EXTEND
-------------
Subclass ``ImageGenClient`` and implement one abstract method:

    ``_generate(request) -> ImageGenResponse``
        Perform the actual provider call and return generated images.

Minimal example
~~~~~~~~~~~~~~~
::

    from conduit_sdk.clients import ImageGenClient
    from conduit_sdk.models.requests import ImageGenRequest
    from conduit_sdk.models.responses import ImageGenResponse, GeneratedImage

    class MyImageClient(ImageGenClient):
        async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
            raw = await my_provider.txt2img(
                prompt=request.prompt,
                width=request.size.width,
                height=request.size.height,
            )
            return ImageGenResponse(
                images=[GeneratedImage(url=raw.image_url)],
                usage=Usage(image_count=1),
            )
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

from conduit_sdk.core.base import BaseClient
from conduit_sdk.core.middleware import CallContext
from conduit_sdk.models.requests import ImageGenRequest
from conduit_sdk.models.responses import ImageGenResponse


class ImageGenClient(BaseClient):
    """Abstract base class for image-generation clients."""

    @abstractmethod
    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        """
        Perform the actual provider call.

        Parameters
        ----------
        request:
            Validated, immutable image-generation request.

        Returns
        -------
        ImageGenResponse
            Must include at least one entry in ``images``.
        """
        raise NotImplementedError

    async def generate(self, request: ImageGenRequest) -> ImageGenResponse:
        """
        Generate images, running the full middleware pipeline first.

        Parameters
        ----------
        request:
            The image generation request to execute.
        """

        async def _handler(ctx: CallContext) -> ImageGenResponse:
            return await self._generate(ctx.request)  # type: ignore[arg-type]

        result = await self._execute(request, _handler)
        return result  # type: ignore[return-value]

    def generate_sync(self, request: ImageGenRequest) -> ImageGenResponse:
        """Synchronous wrapper — runs the event loop for you."""
        return asyncio.run(self.generate(request))
