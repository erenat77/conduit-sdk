"""
Example 02 — LLM streaming
===========================
Shows:
  - Using client.stream() to receive token-by-token deltas
  - Collecting chunks into a full string
  - Real-time printing with flush
  - How streaming differs from generate() (no cost/usage on stream chunks)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from conduit_sdk.clients import LLMClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason, LLMResponse

# ---------------------------------------------------------------------------
# Mock streaming provider — simulates word-by-word delivery with a delay
# ---------------------------------------------------------------------------

MOCK_REPLY = (
    "Streaming works by sending small chunks of the response as they are generated, "
    "rather than waiting for the complete output. This makes your UI feel much faster "
    "because the user sees text appearing immediately."
)


class MockStreamingClient(LLMClient):
    async def _generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=MOCK_REPLY),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=20, completion_tokens=40, total_tokens=60),
            model=self.config.model,
            provider=self.config.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Simulate streaming by yielding one word at a time with a small delay."""
        words = MOCK_REPLY.split()
        for i, word in enumerate(words):
            # Add space between words; newline every 12 words for readability
            suffix = "\n" if (i + 1) % 12 == 0 else " "
            yield word + suffix
            await asyncio.sleep(0.05)  # simulate network latency per token


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

client = MockStreamingClient(
    config=ClientConfig(provider="mock", model="mock-stream-v1"),
    middleware=MiddlewarePipeline([]),  # bare pipeline — no extra overhead
)


async def main() -> None:
    print("=" * 55)
    print("Example 02 — Streaming LLM")
    print("=" * 55)

    request = LLMRequest(
        messages=[Message.user("How does streaming work?")],
        stream=True,
    )

    # --- Option A: print chunks in real time ---
    print("\n[Streaming in real time]\n")
    collected: list[str] = []

    async for chunk in client.stream(request):
        print(chunk, end="", flush=True)
        collected.append(chunk)

    full_text = "".join(collected)

    print(f"\n\n[Complete text assembled from {len(collected)} chunks]")
    print(f"Total characters: {len(full_text)}")
    print(f"Total words     : {len(full_text.split())}")

    # --- Option B: compare with non-streaming ---
    print("\n[Non-streaming generate() for comparison]\n")
    response = await client.generate(request)
    print(response.content)
    print(f"\nUsage: {response.usage.total_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
    # Note: client.generate_sync() can be used here (outside the event loop)
