"""
LLMClient — abstract base for large-language-model clients.

HOW TO EXTEND
-------------
Subclass ``LLMClient`` and implement exactly **two** abstract methods:

    ``_generate(request) -> LLMResponse``
        The actual provider call for a non-streaming completion.

    ``_stream(request) -> AsyncIterator[str]``
        Yield raw text delta chunks from the provider's streaming endpoint.

Everything else (retry, rate-limiting, cost tracking, logging) is handled
automatically by the middleware pipeline inherited from ``BaseClient``.

Minimal example
~~~~~~~~~~~~~~~
::

    from conduit_sdk.clients import LLMClient
    from conduit_sdk.models.requests import LLMRequest
    from conduit_sdk.models.responses import LLMResponse
    from conduit_sdk.models.common import Message, MessageRole, Usage

    class EchoLLMClient(LLMClient):
        \"\"\"Trivial echo client for testing.\"\"\"

        async def _generate(self, request: LLMRequest) -> LLMResponse:
            last_msg = request.messages[-1].content
            return LLMResponse(
                message=Message(role=MessageRole.ASSISTANT, content=last_msg),
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
            for word in request.messages[-1].content.split():
                yield word + " "
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import AsyncIterator

from conduit_sdk.core.base import BaseClient
from conduit_sdk.core.middleware import CallContext
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import LLMResponse


class LLMClient(BaseClient):
    """Abstract base class for LLM (chat completion) clients."""

    # ------------------------------------------------------------------
    # Abstract hooks — implement these in your provider subclass
    # ------------------------------------------------------------------

    @abstractmethod
    async def _generate(self, request: LLMRequest) -> LLMResponse:
        """
        Perform the actual provider call and return a complete response.

        This method is invoked *inside* the middleware pipeline, so retry,
        rate-limit, and logging have already been applied when it runs.

        Parameters
        ----------
        request:
            Validated, immutable request object.

        Returns
        -------
        LLMResponse
            Must include at least ``message`` and ideally ``usage``.
        """
        raise NotImplementedError

    @abstractmethod
    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Yield raw text delta strings from the provider's streaming endpoint.

        Parameters
        ----------
        request:
            Validated, immutable request object.

        Yields
        ------
        str
            A chunk of the assistant's reply (may be a single token, a word,
            or a sentence depending on the provider).
        """
        raise NotImplementedError
        yield  # make type-checker happy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Send a chat completion request and return the full response.

        Runs the middleware pipeline (logging → rate-limit → retry → cost)
        before delegating to ``_generate``.

        Parameters
        ----------
        request:
            The LLM request to execute.
        """

        async def _handler(ctx: CallContext) -> LLMResponse:
            return await self._generate(ctx.request)  # type: ignore[arg-type]

        result = await self._execute(request, _handler)
        return result  # type: ignore[return-value]

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Send a streaming chat request and yield text delta chunks.

        Note: The middleware pipeline runs *before* the first token is yielded.
        Retry and rate-limiting apply to the initial connection; individual
        chunks are not retried.

        Usage::

            async for chunk in client.stream(request):
                print(chunk, end="", flush=True)
        """

        # Run pre-flight middleware with a sentinel response so logging / rate
        # limiting fire before we open the stream.
        async def _stream_handler(ctx: CallContext) -> AsyncIterator[str]:  # type: ignore[return]
            return self._stream(ctx.request)  # type: ignore[arg-type]

        iterator = await self._execute(request, _stream_handler)
        async for chunk in iterator:  # type: ignore[union-attr]
            yield chunk

    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper — runs the event loop for you."""
        return asyncio.run(self.generate(request))
