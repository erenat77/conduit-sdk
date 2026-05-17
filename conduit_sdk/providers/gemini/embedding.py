"""
GeminiEmbeddingClient — adapter for Google Gemini Embeddings API.

Supported models: text-embedding-004, embedding-001

Install: pip install llm-conduit[gemini]
"""

from __future__ import annotations

import os
from typing import Any

from conduit_sdk.clients.embedding import EmbeddingClient
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import EmbeddingRequest
from conduit_sdk.models.responses import Embedding, EmbeddingResponse
from conduit_sdk.providers.gemini.llm import _require_genai, _wrap_genai_error

# Gemini task types for embeddings
_TASK_TYPE_MAP = {
    "query": "RETRIEVAL_QUERY",
    "document": "RETRIEVAL_DOCUMENT",
    "image": "RETRIEVAL_DOCUMENT",  # fall back to document
}


class GeminiEmbeddingClient(EmbeddingClient):
    """
    Embedding client adapter for Google Gemini Embeddings API.

    Supported models:
      - ``text-embedding-004``   — 768-dim, current recommended model
      - ``embedding-001``        — legacy 768-dim model

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``GOOGLE_API_KEY`` environment variable

    Example::

        from conduit_sdk.providers.gemini import GeminiEmbeddingClient
        from conduit_sdk.core.config import ClientConfig

        client = GeminiEmbeddingClient(ClientConfig(
            provider="gemini",
            model="text-embedding-004",
            api_key="AIza...",
        ))
        response = await client.embed(EmbeddingRequest(
            inputs=["The Eiffel Tower is in Paris"],
            input_type="document",
        ))
        print(len(response.vectors[0]))   # 768
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("GOOGLE_API_KEY", "")

    def _sdk_client(self):
        genai = _require_genai()
        return genai.Client(api_key=self._api_key())

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        client = self._sdk_client()
        model = request.model or self.config.model

        task_type = _TASK_TYPE_MAP.get(request.input_type or "document", "RETRIEVAL_DOCUMENT")

        kwargs: dict[str, Any] = {"task_type": task_type}
        if request.dimensions is not None:
            kwargs["output_dimensionality"] = request.dimensions
        kwargs.update(request.extra)

        try:
            raw = await client.aio.models.embed_content(
                model=model,
                contents=request.inputs,
                config=kwargs,
            )
        except Exception as exc:
            raise _wrap_genai_error(exc) from exc

        vectors = [e.values for e in raw.embeddings]

        return EmbeddingResponse(
            embeddings=[
                Embedding(index=i, vector=v) for i, v in enumerate(vectors)
            ],
            usage=Usage(
                prompt_tokens=0,
                total_tokens=0,
                embedding_count=len(request.inputs),
            ),
            model=model,
            provider="gemini",
            raw_response=raw,
        )
