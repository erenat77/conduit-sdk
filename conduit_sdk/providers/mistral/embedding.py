"""
MistralEmbeddingClient — adapter for Mistral AI Embeddings API.

Supported models: mistral-embed

Install: pip install llm-conduit[mistral]
"""

from __future__ import annotations

import os
from typing import Any

from conduit_sdk.clients.embedding import EmbeddingClient
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import EmbeddingRequest
from conduit_sdk.models.responses import Embedding, EmbeddingResponse
from conduit_sdk.providers.mistral.llm import _require_mistral, _wrap_mistral_error


class MistralEmbeddingClient(EmbeddingClient):
    """
    Embedding client adapter for Mistral AI's Embeddings API.

    Supported models:
      - ``mistral-embed``   — 1024-dim, optimised for retrieval

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``MISTRAL_API_KEY`` environment variable

    Example::

        from conduit_sdk.providers.mistral import MistralEmbeddingClient
        from conduit_sdk.core.config import ClientConfig

        client = MistralEmbeddingClient(ClientConfig(
            provider="mistral",
            model="mistral-embed",
            api_key="...",
        ))
        response = await client.embed(EmbeddingRequest(
            inputs=["Attention is all you need", "Transformers changed NLP"],
        ))
        print(len(response.vectors))   # 2
        print(len(response.vectors[0]))  # 1024
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("MISTRAL_API_KEY", "")

    def _sdk_client(self):
        mistralai = _require_mistral()
        return mistralai.Mistral(
            api_key=self._api_key(),
            timeout_ms=int(self.config.timeout_seconds * 1000),
        )

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = request.model or self.config.model
        kwargs: dict[str, Any] = {
            "model": model,
            "inputs": request.inputs,
        }
        kwargs.update(request.extra)

        try:
            raw = await self._sdk_client().embeddings.create_async(**kwargs)
        except Exception as exc:
            raise _wrap_mistral_error(exc) from exc

        return EmbeddingResponse(
            embeddings=[Embedding(index=item.index, vector=item.embedding) for item in raw.data],
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
                total_tokens=raw.usage.total_tokens if raw.usage else 0,
                embedding_count=len(request.inputs),
            ),
            model=model,
            provider="mistral",
            raw_response=raw,
        )
