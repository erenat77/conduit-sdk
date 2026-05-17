"""
Example 06 — Cost tracking across a session
=============================================
Shows:
  - CostConfig with per-token / per-image / per-second pricing
  - CostTracker shared across multiple client calls
  - Per-call cost on response.cost
  - Session summary after N calls
  - Resetting the tracker
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from conduit_sdk.clients import EmbeddingClient, ImageGenClient, LLMClient
from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import EmbeddingRequest, ImageGenRequest, LLMRequest
from conduit_sdk.models.responses import (
    Embedding,
    EmbeddingResponse,
    FinishReason,
    GeneratedImage,
    ImageGenResponse,
    LLMResponse,
)
from conduit_sdk.utils.cost import CostMiddleware, CostTracker

# ---------------------------------------------------------------------------
# Mock clients that return realistic usage stats
# ---------------------------------------------------------------------------


class MockLLM(LLMClient):
    async def _generate(self, request: LLMRequest) -> LLMResponse:
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = 150  # simulate a medium-length reply
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content="A detailed response " * 15),
            finish_reason=FinishReason.STOP,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=self.config.model,
            provider=self.config.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        yield "mock"


class MockImageGen(ImageGenClient):
    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        return ImageGenResponse(
            images=[
                GeneratedImage(url="https://example.com/img.png") for _ in range(request.num_images)
            ],
            usage=Usage(image_count=request.num_images),
            model=self.config.model,
            provider=self.config.provider,
        )


class MockEmbedder(EmbeddingClient):
    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        tokens = sum(len(t.split()) for t in request.inputs)
        return EmbeddingResponse(
            embeddings=[
                Embedding(index=i, vector=[0.1, 0.2, 0.3]) for i in range(len(request.inputs))
            ],
            usage=Usage(
                prompt_tokens=tokens, total_tokens=tokens, embedding_count=len(request.inputs)
            ),
            model=self.config.model,
            provider=self.config.provider,
        )


# ---------------------------------------------------------------------------
# Build clients with pricing and a shared tracker
# ---------------------------------------------------------------------------

tracker = CostTracker()


def _make_pipeline(cost_cfg: CostConfig) -> MiddlewarePipeline:
    return MiddlewarePipeline([CostMiddleware(cost_cfg, tracker=tracker)])


llm_client = MockLLM(
    config=ClientConfig(
        provider="openai",
        model="gpt-4o",
        cost=CostConfig(input_cost_per_1k_tokens=0.005, output_cost_per_1k_tokens=0.015),
    ),
    middleware=_make_pipeline(
        CostConfig(input_cost_per_1k_tokens=0.005, output_cost_per_1k_tokens=0.015)
    ),
)

image_client = MockImageGen(
    config=ClientConfig(
        provider="openai", model="dall-e-3", cost=CostConfig(image_cost_per_unit=0.04)
    ),
    middleware=_make_pipeline(CostConfig(image_cost_per_unit=0.04)),
)

embed_client = MockEmbedder(
    config=ClientConfig(
        provider="openai",
        model="text-embedding-3-large",
        cost=CostConfig(embedding_cost_per_1k_tokens=0.00013),
    ),
    middleware=_make_pipeline(CostConfig(embedding_cost_per_1k_tokens=0.00013)),
)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 55)
    print("Example 06 — Session Cost Tracking")
    print("=" * 55)

    # --- Several LLM calls ---
    print("\n[LLM calls × 3]")
    prompts = [
        "Summarize the history of the Roman Empire.",
        "Explain quantum entanglement in simple terms.",
        "Write a haiku about machine learning.",
    ]
    for prompt in prompts:
        req = LLMRequest(messages=[Message.user(prompt)])
        resp = await llm_client.generate(req)
        print(f"  ✓ {prompt[:45]!r}…  → ${resp.cost.total_cost:.5f}")

    # --- Image generation ---
    print("\n[Image generation × 2 calls]")
    for prompt in ["A futuristic robot", "A tranquil forest"]:
        resp = await image_client.generate(ImageGenRequest(prompt=prompt, num_images=2))
        print(f"  ✓ {prompt!r} × {resp.usage.image_count} imgs  → ${resp.cost.total_cost:.4f}")

    # --- Embeddings ---
    print("\n[Embeddings — batch of 5]")
    resp = await embed_client.embed(
        EmbeddingRequest(
            inputs=[
                "The Eiffel Tower",
                "Machine learning",
                "Python",
                "The Louvre",
                "Neural networks",
            ]
        )
    )
    print(f"  ✓ 5 inputs, {resp.usage.total_tokens} tokens  → ${resp.cost.total_cost:.6f}")

    # --- Session summary ---
    summary = tracker.summary()
    print("\n" + "=" * 55)
    print("SESSION SUMMARY")
    print("=" * 55)
    print(f"  Total calls  : {summary['call_count']}")
    print(f"  Total tokens : {summary['total_tokens']}")
    print(f"  Total cost   : ${summary['total_cost_usd']:.4f} USD")

    # --- Reset for next session ---
    tracker.reset()
    print(f"\n[After reset] call_count = {tracker.call_count}")


if __name__ == "__main__":
    asyncio.run(main())
