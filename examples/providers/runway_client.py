"""
Provider skeleton — Runway
===========================
Covers: VideoGenClient (Runway Gen-3 Alpha / Gen-3 Alpha Turbo).

Install: pip install runwayml
"""

from __future__ import annotations

import asyncio

from conduit_sdk.clients import VideoGenClient
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import VideoGenRequest
from conduit_sdk.models.responses import GeneratedVideo, VideoGenResponse


class RunwayVideoClient(VideoGenClient):
    """
    Video-generation adapter for Runway's Gen-3 models.

    Runway uses an async job system — this adapter submits a task and polls
    until it reaches SUCCEEDED or FAILED status.

    Usage::

        client = RunwayVideoClient(ClientConfig(
            provider="runway",
            model="gen3a_turbo",
            api_key="...",
        ))
        response = await client.generate(VideoGenRequest(
            prompt="A drone flyover of a coastal city at dusk",
            reference_image_url="https://example.com/frame.jpg",  # required for image-to-video
            duration_seconds=5.0,
            fps=24,
        ))
        print(response.first.url)
    """

    POLL_INTERVAL_SECONDS: float = 5.0
    MAX_POLL_ATTEMPTS: int = 60  # 5 min at 5s intervals

    def _sdk(self):
        import runwayml  # noqa: PLC0415

        return runwayml.AsyncRunwayML(api_key=self.config.api_key)

    async def _generate(self, request: VideoGenRequest) -> VideoGenResponse:
        sdk = self._sdk()

        # Runway requires an image URL for image-to-video
        if not request.reference_image_url:
            raise ValueError(
                "RunwayVideoClient requires a reference_image_url. "
                "Runway Gen-3 is an image-to-video model."
            )

        # Map duration: Runway supports 5s or 10s
        duration = 10 if request.duration_seconds > 5 else 5

        task = await sdk.image_to_video.create(
            model=self.config.model,
            prompt_image=request.reference_image_url,
            prompt_text=request.prompt,
            duration=duration,
            ratio="16:9",
            **request.extra,
        )

        task_id = task.id

        # Poll until complete
        for _ in range(self.MAX_POLL_ATTEMPTS):
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
            task = await sdk.tasks.retrieve(task_id)

            if task.status == "SUCCEEDED":
                video_url = task.output[0] if task.output else None
                return VideoGenResponse(
                    videos=[
                        GeneratedVideo(
                            url=video_url,
                            duration_seconds=float(duration),
                            fps=request.fps,
                        )
                    ],
                    usage=Usage(video_seconds=float(duration)),
                    model=self.config.model,
                    provider="runway",
                    raw_response=task,
                )

            if task.status == "FAILED":
                from conduit_sdk.core.exceptions import ProviderError

                raise ProviderError(
                    f"Runway task {task_id} failed: {getattr(task, 'failure', 'unknown error')}",
                    provider="runway",
                )

        from conduit_sdk.core.exceptions import TimeoutError as SDKTimeout

        raise SDKTimeout(
            f"Runway task {task_id} did not complete within "
            f"{self.MAX_POLL_ATTEMPTS * self.POLL_INTERVAL_SECONDS:.0f}s",
            provider="runway",
        )
