"""
CohereEmbeddingClient — adapter for Cohere Embeddings API v2.

Supported models: embed-v4.0, embed-english-v3.0, embed-multilingual-v3.0, …

Install: pip install llm-conduit[cohere]
"""

from __future__ import annotations

import os
from typing import Any

from conduit_sdk.clients.embedding import EmbeddingClient
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import EmbeddingRequest
from conduit_sdk.models.responses import Embedding, EmbeddingResponse
from conduit_sdk.providers.cohere.llm import _require_cohere, _wrap_cohere_error

# Cohere input type mapping
_INPUT_TYPE_MAP = {
    "query": "search_query",
    "document": "search_document",
    "image": "image",
}


class CohereEmbeddingClient(EmbeddingClient):
    """
    Embedding client adapter for Cohere's Embed API (v2).

    Supported models:
      - ``embed-v4.0``                    — latest, 1536-dim (text + image)
      - ``embed-english-v3.0``            — English, 1024-dim
      - ``embed-multilingual-v3.0``       — 100+ languages, 1024-dim

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``COHERE_API_KEY`` environment variable

    Example::

        from conduit_sdk.providers.cohere import CohereEmbeddingClient
        from conduit_sdk.core.config import ClientConfig

        client = CohereEmbeddingClient(ClientConfig(
            provider="cohere",
            model="embed-english-v3.0",
            api_key="...",
        ))
        response = await client.embed(EmbeddingRequest(
            inputs=["The Eiffel Tower is in Paris", "Machine learning is fun"],
            input_type="document",
        ))
        print(len(response.vectors))   # 2
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("COHERE_API_KEY", "")

    def _sdk_client(self):
        cohere = _require_cohere()
        return cohere.AsyncClientV2(
            api_key=self._api_key(),
            timeout=self.config.timeout_seconds,
        )

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = request.model or self.config.model
        input_type = _INPUT_TYPE_MAP.get(request.input_type or "document", "search_document")

        kwargs: dict[str, Any] = {
            "model": model,
            "texts": request.inputs,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        if request.dimensions is not None:
            kwargs["output_dimension"] = request.dimensions
        kwargs.update(request.extra)

        try:
            raw = await self._sdk_client().embed(**kwargs)
        except Exception as exc:
            raise _wrap_cohere_error(exc) from exc

        # v2 response: raw.embeddings.float_ is the list of float vectors
        vectors = raw.embeddings.float_ or []

        return EmbeddingResponse(
            embeddings=[Embedding(index=i, vector=v) for i, v in enumerate(vectors)],
            usage=Usage(
                prompt_tokens=0,
                total_tokens=0,
                embedding_count=len(request.inputs),
            ),
            model=model,
            provider="cohere",
            raw_response=raw,
        )
