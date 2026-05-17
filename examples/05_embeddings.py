"""
Example 05 — Embeddings + cosine similarity
============================================
Shows:
  - Implementing EmbeddingClient
  - Batch embedding multiple inputs in one call
  - Cosine similarity between vectors (retrieval use case)
  - input_type hint for asymmetric retrieval (query vs document)
  - Accessing response fields: vectors, dimensions, usage
"""

from __future__ import annotations

import asyncio
import math
import random

from conduit_sdk.clients import EmbeddingClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import EmbeddingRequest
from conduit_sdk.models.responses import Embedding, EmbeddingResponse

# ---------------------------------------------------------------------------
# Mock embedding provider — returns deterministic pseudo-random vectors
# ---------------------------------------------------------------------------

DIMS = 8  # small for readability; real models use 768–3072


def _text_to_vector(text: str, dims: int) -> list[float]:
    """Deterministic pseudo-random unit vector seeded by text hash."""
    rng = random.Random(hash(text) % (2**32))
    vec = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(v**2 for v in vec))
    return [v / norm for v in vec]


class MockEmbeddingClient(EmbeddingClient):
    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        dims = request.dimensions or DIMS
        embeddings = [
            Embedding(index=i, vector=_text_to_vector(text, dims))
            for i, text in enumerate(request.inputs)
        ]
        tokens = sum(len(t.split()) for t in request.inputs)
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=Usage(
                prompt_tokens=tokens,
                total_tokens=tokens,
                embedding_count=len(request.inputs),
            ),
            model=self.config.model,
            provider=self.config.provider,
        )


# ---------------------------------------------------------------------------
# Utility: cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(y**2 for y in b))
    return dot / (norm_a * norm_b + 1e-10)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

client = MockEmbeddingClient(
    config=ClientConfig(provider="mock", model="mock-embed-v1"),
    middleware=MiddlewarePipeline([]),
)


async def main() -> None:
    print("=" * 55)
    print("Example 05 — Embeddings + Cosine Similarity")
    print("=" * 55)

    # --- Batch embed documents ---
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The Louvre Museum houses the Mona Lisa painting.",
        "Neural networks are inspired by biological brains.",
    ]

    print(f"\n[Embedding {len(documents)} documents]")
    doc_response = await client.embed(
        EmbeddingRequest(
            inputs=documents,
            input_type="document",
        )
    )

    print(f"  Dimensions : {doc_response.dimensions}")
    print(f"  Embeddings : {len(doc_response.embeddings)}")
    print(f"  Tokens used: {doc_response.usage.total_tokens}")

    # --- Embed a query and rank documents by similarity ---
    query = "What famous artwork is in a French museum?"

    print(f"\n[Query] '{query}'")
    query_response = await client.embed(
        EmbeddingRequest(
            inputs=[query],
            input_type="query",
        )
    )

    query_vec = query_response.vectors[0]
    doc_vecs = doc_response.vectors

    scores = [
        (cosine_similarity(query_vec, doc_vec), doc)
        for doc_vec, doc in zip(doc_vecs, documents, strict=True)
    ]
    scores.sort(reverse=True)

    print("\n[Ranked results by cosine similarity]")
    for rank, (score, doc) in enumerate(scores, 1):
        print(f"  {rank}. ({score:+.4f})  {doc}")


if __name__ == "__main__":
    asyncio.run(main())

    print("\n[Sync embed call]")
    sync_resp = client.embed_sync(EmbeddingRequest(inputs=["Hello world"]))
    print(f"  Vector[:4]: {sync_resp.vectors[0][:4]}")
    print(f"  Dimensions: {sync_resp.dimensions}")
