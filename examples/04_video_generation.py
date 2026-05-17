"""
Example 04 — Video generation
==============================
Shows:
  - Implementing VideoGenClient (including async job polling pattern)
  - Using duration, fps, resolution fields
  - Accessing GeneratedVideo metadata
  - Pattern for providers that return a job ID and require polling
"""

from __future__ import annotations

import asyncio

from conduit_sdk.clients import VideoGenClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import ImageSize, VideoGenRequest
from conduit_sdk.models.responses import GeneratedVideo, VideoGenResponse

# ---------------------------------------------------------------------------
# Mock video provider — simulates an async job system (submit → poll → result)
# ---------------------------------------------------------------------------


class MockVideoClient(VideoGenClient):
    """
    Simulates a video generation provider with async job polling.

    Many real video providers (Runway, Kling, Sora) work this way:
      1. Submit a job → get a job ID
      2. Poll until status == SUCCEEDED
      3. Return the output URL

    This mock skips the actual wait but shows the pattern.
    """

    async def _submit_job(self, request: VideoGenRequest) -> str:
        """Simulate submitting a job and returning a job ID."""
        await asyncio.sleep(0.01)  # simulated network round-trip
        return f"job_mock_{hash(request.prompt) % 100000:05d}"

    async def _poll_job(self, job_id: str, request: VideoGenRequest) -> str:
        """Simulate polling until the job completes."""
        for _attempt in range(3):
            await asyncio.sleep(0.01)  # in production: sleep(5) between polls
            # Pretend it succeeds on the last attempt
        return (
            f"https://mock-cdn.example.com/videos/{job_id}"
            f"_{int(request.duration_seconds)}s_{request.fps}fps.mp4"
        )

    async def _generate(self, request: VideoGenRequest) -> VideoGenResponse:
        # Step 1: submit
        job_id = await self._submit_job(request)
        print(f"  Job submitted: {job_id}")

        # Step 2: poll
        video_url = await self._poll_job(job_id, request)
        print(f"  Job complete : {video_url}")

        return VideoGenResponse(
            videos=[
                GeneratedVideo(
                    url=video_url,
                    duration_seconds=request.duration_seconds,
                    fps=request.fps,
                    width=request.resolution.width,
                    height=request.resolution.height,
                    seed=request.seed,
                    mime_type="video/mp4",
                )
            ],
            usage=Usage(video_seconds=request.duration_seconds),
            model=self.config.model,
            provider=self.config.provider,
        )


client = MockVideoClient(
    config=ClientConfig(provider="mock", model="mock-video-gen-v2"),
    middleware=MiddlewarePipeline([]),
)


async def main() -> None:
    print("=" * 55)
    print("Example 04 — Video Generation")
    print("=" * 55)

    # --- Short clip ---
    print("\n[Short clip — 4s at 24fps]")
    response = await client.generate(
        VideoGenRequest(
            prompt="A time-lapse of clouds rolling over a mountain range at sunset",
            duration_seconds=4.0,
            fps=24,
            resolution=ImageSize(width=1280, height=720),
            seed=777,
        )
    )

    video = response.first
    print(f"  URL       : {video.url}")
    print(f"  Duration  : {video.duration_seconds}s")
    print(f"  FPS       : {video.fps}")
    print(f"  Resolution: {video.width}×{video.height}")
    print(f"  Seed      : {video.seed}")

    # --- Image-to-video ---
    print("\n[Image-to-video — using a reference image]")
    img2vid = await client.generate(
        VideoGenRequest(
            prompt="Gentle waves lapping on a sandy beach",
            reference_image_url="https://example.com/beach.jpg",
            duration_seconds=6.0,
            fps=30,
        )
    )
    print(f"  Video URL : {img2vid.first.url}")
    print(f"  Duration  : {img2vid.first.duration_seconds}s")

    print(f"\nTotal video seconds generated: {response.usage.video_seconds}")


if __name__ == "__main__":
    asyncio.run(main())

    print("\n[Sync call]")
    sync_resp = client.generate_sync(
        VideoGenRequest(
            prompt="Stars rotating above a desert, timelapse",
            duration_seconds=5.0,
        )
    )
    print(f"  Sync video: {sync_resp.first.url}")
