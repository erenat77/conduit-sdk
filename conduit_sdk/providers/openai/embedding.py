"""
OpenAIEmbeddingClient — adapter for OpenAI Embeddings API.

Supported models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
"""

from __future__ import annotations

import os
from typing import Any

from conduit_sdk.clients.embedding import EmbeddingClient
from conduit_sdk.models.common import Usage
from conduit_sdk.models.requests import EmbeddingRequest
from conduit_sdk.models.responses import Embedding, EmbeddingResponse
from conduit_sdk.providers.openai.llm import _require_openai, _wrap_openai_error


class OpenAIEmbeddingClient(EmbeddingClient):
    """
    Embedding client adapter for OpenAI's Embeddings API.

    Supported models:
      - ``text-embedding-3-large``  — 3072 dims, best quality
      - ``text-embedding-3-small``  — 1536 dims, faster and cheaper
      - ``text-embedding-ada-002``  — legacy, 1536 dims

    Supports Matryoshka Representation Learning (MRL) via ``request.dimensions``
    to reduce vector size without retraining (3-large and 3-small only).

    Example::

        client = OpenAIEmbeddingClient(ClientConfig(
            model="text-embedding-3-large",
            api_key="sk-...",
        ))
        response = await client.embed(EmbeddingRequest(
            inputs=["The Eiffel Tower is in Paris", "Machine learning is fun"],
            input_type="document",
            dimensions=512,       # MRL: reduce from 3072 → 512
        ))
        print(response.dimensions)   # 512
        print(len(response.vectors)) # 2
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("OPENAI_API_KEY", "")

    def _sdk_client(self):
        openai = _require_openai()
        return openai.AsyncOpenAI(
            api_key=self._api_key(),
            base_url=self.config.api_base_url or None,
            timeout=self.config.timeout_seconds,
        )

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.config.model,
            "input": request.inputs,
            "encoding_format": request.encoding_format,
        }
        if request.dimensions is not None:
            kwargs["dimensions"] = request.dimensions
        kwargs.update(request.extra)

        try:
            raw = await self._sdk_client().embeddings.create(**kwargs)
        except Exception as exc:
            raise _wrap_openai_error(exc) from exc

        return EmbeddingResponse(
            embeddings=[Embedding(index=item.index, vector=item.embedding) for item in raw.data],
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                total_tokens=raw.usage.total_tokens,
                embedding_count=len(request.inputs),
            ),
            model=raw.model,
            provider="openai",
            raw_response=raw,
        )
