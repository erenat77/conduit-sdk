"""
ReplicateVideoClient — adapter for Replicate video generation models.

Uses the official ``replicate`` Python SDK.

Supported models (pass as config.model or request.model):
  - "minimax/video-01"
  - "luma-ai/dream-machine"
  - "stability-ai/stable-video-diffusion"
  - any Replicate video model version string

Install: pip install llm-conduit[replicate]
"""

from __future__ import annotations

import os
from typing import Any

from conduit_sdk.clients.video import VideoGenClient
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import VideoGenRequest
from conduit_sdk.models.responses import GeneratedVideo, VideoGenResponse
from conduit_sdk.providers.replicate.image import _require_replicate, _wrap_replicate_error


class ReplicateVideoClient(VideoGenClient):
    """
    Video generation client adapter for Replicate.

    Replicate hosts video generation models that may take minutes to complete.
    The ``replicate`` SDK handles polling automatically — ``async_run`` waits
    until the prediction is finished before returning.

    The API token is resolved in this order:
      1. ``config.api_key``
      2. ``REPLICATE_API_TOKEN`` environment variable

    Example::

        from conduit_sdk.providers.replicate import ReplicateVideoClient
        from conduit_sdk.core.config import ClientConfig

        client = ReplicateVideoClient(ClientConfig(
            provider="replicate",
            model="minimax/video-01",
            api_key="r8_...",   # or set REPLICATE_API_TOKEN env var
        ))
        response = await client.generate(VideoGenRequest(
            prompt="A drone shot of a mountain range at golden hour",
            duration_seconds=5.0,
        ))
        print(response.videos[0].url)
    """

    def _api_token(self) -> str:
        return self.config.api_key or os.environ.get("REPLICATE_API_TOKEN", "")

    def _sdk_client(self):
        replicate = _require_replicate()
        return replicate.Client(api_token=self._api_token())

    async def _generate(self, request: VideoGenRequest) -> VideoGenResponse:
        model = request.model or self.config.model

        # Build input dict — Replicate model inputs vary by model,
        # but these are the common fields across video models.
        input_data: dict[str, Any] = {"prompt": request.prompt}
        if request.duration_seconds is not None:
            input_data["duration"] = request.duration_seconds
        if request.fps is not None:
            input_data["fps"] = request.fps
        if request.resolution is not None:
            input_data["width"] = request.resolution.width
            input_data["height"] = request.resolution.height
        if request.reference_image_url is not None:
            input_data["image"] = request.reference_image_url
        if request.seed is not None:
            input_data["seed"] = request.seed
        input_data.update(request.extra)

        try:
            client = self._sdk_client()
            output = await client.async_run(model, input=input_data)
        except Exception as exc:
            raise _wrap_replicate_error(exc) from exc

        # Replicate returns a URL string or list of URLs
        video_urls: list[str] = []
        if isinstance(output, list):
            video_urls = [str(item) for item in output]
        elif output:
            video_urls = [str(output)]

        videos = [
            GeneratedVideo(
                url=url,
                duration_seconds=request.duration_seconds,
                fps=request.fps,
            )
            for url in video_urls
        ]

        total_seconds = sum(v.duration_seconds or 0.0 for v in videos)
        return VideoGenResponse(
            videos=videos,
            usage=Usage(video_seconds=total_seconds),
            model=model,
            provider="replicate",
            raw_response=output,
        )
