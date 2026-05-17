"""
EmbeddingClient — abstract base for text/multi-modal embedding clients.

HOW TO EXTEND
-------------
Subclass ``EmbeddingClient`` and implement one abstract method:

    ``_embed(request) -> EmbeddingResponse``
        Encode the inputs and return dense vectors.

Minimal example
~~~~~~~~~~~~~~~
::

    from conduit_sdk.clients import EmbeddingClient
    from conduit_sdk.models.requests import EmbeddingRequest
    from conduit_sdk.models.responses import EmbeddingResponse, Embedding
    from conduit_sdk.models.common import Usage

    class MyEmbeddingClient(EmbeddingClient):
        async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
            raw = await my_provider.embed(texts=request.inputs)
            return EmbeddingResponse(
                embeddings=[
                    Embedding(index=i, vector=vec)
                    for i, vec in enumerate(raw.vectors)
                ],
                usage=Usage(
                    prompt_tokens=raw.tokens_used,
                    total_tokens=raw.tokens_used,
                    embedding_count=len(request.inputs),
                ),
            )
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

from conduit_sdk.core.base import BaseClient
from conduit_sdk.core.middleware import CallContext
from conduit_sdk.models.requests import EmbeddingRequest
from conduit_sdk.models.responses import EmbeddingResponse


class EmbeddingClient(BaseClient):
    """Abstract base class for embedding clients."""

    @abstractmethod
    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Perform the actual embedding call.

        Parameters
        ----------
        request:
            Validated, immutable embedding request.

        Returns
        -------
        EmbeddingResponse
            Must include one ``Embedding`` per input, in the same order.
        """
        raise NotImplementedError

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Encode inputs into embedding vectors, running the middleware pipeline first.

        Parameters
        ----------
        request:
            The embedding request to execute.
        """

        async def _handler(ctx: CallContext) -> EmbeddingResponse:
            return await self._embed(ctx.request)  # type: ignore[arg-type]

        result = await self._execute(request, _handler)
        return result  # type: ignore[return-value]

    def embed_sync(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Synchronous wrapper — runs the event loop for you."""
        return asyncio.run(self.embed(request))
