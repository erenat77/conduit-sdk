"""
VideoGenClient — abstract base for video-generation clients.

HOW TO EXTEND
-------------
Subclass ``VideoGenClient`` and implement one abstract method:

    ``_generate(request) -> VideoGenResponse``
        Perform the actual provider call and return generated video clips.

Many video providers are asynchronous job systems (poll for completion).
Handle that inside ``_generate``; from the SDK's perspective the method
must eventually return a completed ``VideoGenResponse``.

Minimal example
~~~~~~~~~~~~~~~
::

    from conduit_sdk.clients import VideoGenClient
    from conduit_sdk.models.requests import VideoGenRequest
    from conduit_sdk.models.responses import VideoGenResponse, GeneratedVideo

    class MyVideoClient(VideoGenClient):
        async def _generate(self, request: VideoGenRequest) -> VideoGenResponse:
            job = await my_provider.submit(prompt=request.prompt)
            video_url = await job.wait()          # poll until ready
            return VideoGenResponse(
                videos=[GeneratedVideo(
                    url=video_url,
                    duration_seconds=request.duration_seconds,
                    fps=request.fps,
                )],
            )
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

from conduit_sdk.core.base import BaseClient
from conduit_sdk.core.middleware import CallContext
from conduit_sdk.models.requests import VideoGenRequest
from conduit_sdk.models.responses import VideoGenResponse


class VideoGenClient(BaseClient):
    """Abstract base class for video-generation clients."""

    @abstractmethod
    async def _generate(self, request: VideoGenRequest) -> VideoGenResponse:
        """
        Perform the actual provider call.

        Parameters
        ----------
        request:
            Validated, immutable video-generation request.

        Returns
        -------
        VideoGenResponse
            Must include at least one entry in ``videos``.
        """
        raise NotImplementedError

    async def generate(self, request: VideoGenRequest) -> VideoGenResponse:
        """
        Generate a video clip, running the full middleware pipeline first.

        Parameters
        ----------
        request:
            The video generation request to execute.
        """

        async def _handler(ctx: CallContext) -> VideoGenResponse:
            return await self._generate(ctx.request)  # type: ignore[arg-type]

        result = await self._execute(request, _handler)
        return result  # type: ignore[return-value]

    def generate_sync(self, request: VideoGenRequest) -> VideoGenResponse:
        """Synchronous wrapper — runs the event loop for you."""
        return asyncio.run(self.generate(request))
