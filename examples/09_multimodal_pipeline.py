"""
Example 09 — Multi-modal pipeline: LLM → Image → Embedding
============================================================
Shows:
  - Chaining multiple modality clients in one workflow
  - Using LLM output as input to image generation (prompt refinement)
  - Embedding the generated image description for retrieval indexing
  - async/await composition across clients
  - A realistic "generate & index" content pipeline

Workflow:
  1. User gives a rough idea
  2. LLM refines it into a polished image prompt
  3. ImageGenClient generates the image
  4. EmbeddingClient encodes the refined prompt for vector search indexing
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass

from conduit_sdk.clients import EmbeddingClient, ImageGenClient, LLMClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import EmbeddingRequest, ImageGenRequest, ImageSize, LLMRequest
from conduit_sdk.models.responses import (
    Embedding,
    EmbeddingResponse,
    FinishReason,
    GeneratedImage,
    ImageGenResponse,
    LLMResponse,
)

# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------


class PromptRefinerLLM(LLMClient):
    """Expands a rough idea into a detailed image generation prompt."""

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        raw_idea = request.messages[-1].content
        refined = (
            f"A stunning, photorealistic render of {raw_idea}, "
            "dramatic lighting, 8K resolution, ultra-detailed, "
            "professional photography, award-winning composition, "
            "vibrant colors, sharp focus."
        )
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=refined),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=20, completion_tokens=40, total_tokens=60),
            model=self.config.model,
            provider=self.config.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        resp = await self._generate(request)
        yield resp.content


class MockImageGen(ImageGenClient):
    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        return ImageGenResponse(
            images=[
                GeneratedImage(
                    url=f"https://cdn.example.com/{hash(request.prompt) % 99999:05d}.png",
                    revised_prompt=request.prompt,
                    seed=42,
                )
            ],
            usage=Usage(image_count=1),
            model=self.config.model,
            provider=self.config.provider,
        )


class MockEmbedder(EmbeddingClient):
    DIMS = 12

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        def vec(text: str) -> list[float]:
            rng = random.Random(hash(text) % (2**32))
            v = [rng.gauss(0, 1) for _ in range(self.DIMS)]
            n = math.sqrt(sum(x**2 for x in v))
            return [x / n for x in v]

        return EmbeddingResponse(
            embeddings=[Embedding(index=i, vector=vec(t)) for i, t in enumerate(request.inputs)],
            usage=Usage(
                prompt_tokens=sum(len(t.split()) for t in request.inputs),
                total_tokens=sum(len(t.split()) for t in request.inputs),
                embedding_count=len(request.inputs),
            ),
            model=self.config.model,
            provider=self.config.provider,
        )


# ---------------------------------------------------------------------------
# Result dataclass for the pipeline output
# ---------------------------------------------------------------------------


@dataclass
class GeneratedAsset:
    raw_idea: str
    refined_prompt: str
    image_url: str
    embedding_vector: list[float]
    token_usage: int
    image_count: int

    def __repr__(self) -> str:
        vec_preview = [round(v, 3) for v in self.embedding_vector[:4]]
        return (
            f"GeneratedAsset(\n"
            f"  raw_idea      = {self.raw_idea!r}\n"
            f"  refined_prompt= {self.refined_prompt[:60]!r}…\n"
            f"  image_url     = {self.image_url}\n"
            f"  embedding[:4] = {vec_preview}\n"
            f"  tokens_used   = {self.token_usage}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class MultiModalPipeline:
    """
    LLM → ImageGen → Embedding content generation pipeline.

    Given a rough idea string, produces a polished image and a vector
    embedding suitable for indexing into a vector store.
    """

    def __init__(
        self,
        llm: LLMClient,
        image_gen: ImageGenClient,
        embedder: EmbeddingClient,
    ) -> None:
        self._llm = llm
        self._image_gen = image_gen
        self._embedder = embedder

    async def run(self, raw_idea: str) -> GeneratedAsset:
        total_tokens = 0

        # Step 1: Refine the prompt with LLM
        print("\n  [1/3] Refining prompt via LLM...")
        llm_resp = await self._llm.generate(
            LLMRequest(
                messages=[
                    Message.system(
                        "You are a creative director specializing in AI image prompts. "
                        "Expand the user's rough idea into a detailed, vivid prompt."
                    ),
                    Message.user(raw_idea),
                ],
                max_tokens=150,
                temperature=0.8,
            )
        )
        refined_prompt = llm_resp.content
        total_tokens += llm_resp.usage.total_tokens
        print(f"  ✓ Refined: {refined_prompt[:70]}…")

        # Step 2: Generate image from refined prompt
        print("\n  [2/3] Generating image...")
        img_resp = await self._image_gen.generate(
            ImageGenRequest(
                prompt=refined_prompt,
                size=ImageSize(width=1024, height=1024),
                steps=30,
                guidance_scale=7.5,
                seed=42,
            )
        )
        image_url = img_resp.first.url
        print(f"  ✓ Image URL: {image_url}")

        # Step 3: Embed the refined prompt for vector indexing
        print("\n  [3/3] Embedding prompt for vector index...")
        embed_resp = await self._embedder.embed(
            EmbeddingRequest(
                inputs=[refined_prompt],
                input_type="document",
            )
        )
        vector = embed_resp.vectors[0]
        total_tokens += embed_resp.usage.total_tokens
        print(f"  ✓ Embedding dims: {embed_resp.dimensions}")

        return GeneratedAsset(
            raw_idea=raw_idea,
            refined_prompt=refined_prompt,
            image_url=image_url,
            embedding_vector=vector,
            token_usage=total_tokens,
            image_count=img_resp.usage.image_count,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

_bare = MiddlewarePipeline([])

pipeline = MultiModalPipeline(
    llm=PromptRefinerLLM(
        config=ClientConfig(provider="mock-llm", model="refiner-v1"),
        middleware=_bare,
    ),
    image_gen=MockImageGen(
        config=ClientConfig(provider="mock-image", model="diffuser-v3"),
        middleware=_bare,
    ),
    embedder=MockEmbedder(
        config=ClientConfig(provider="mock-embed", model="embed-v2"),
        middleware=_bare,
    ),
)


async def main() -> None:
    print("=" * 55)
    print("Example 09 — Multi-Modal Pipeline")
    print("=" * 55)

    ideas = [
        "a lone astronaut on Mars at sunrise",
        "an ancient library filled with glowing books",
        "a cyberpunk street market in Tokyo at night",
    ]

    assets: list[GeneratedAsset] = []

    for idea in ideas:
        print(f"\n{'─' * 55}")
        print(f"Idea: {idea!r}")
        asset = await pipeline.run(idea)
        assets.append(asset)

    print(f"\n{'=' * 55}")
    print("RESULTS")
    print(f"{'=' * 55}")
    for asset in assets:
        print(f"\n{asset}")

    print(f"\nTotal assets generated : {len(assets)}")
    print(f"Total tokens consumed  : {sum(a.token_usage for a in assets)}")


if __name__ == "__main__":
    asyncio.run(main())
