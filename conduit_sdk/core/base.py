"""
BaseClient — the root of the client hierarchy.

Design
------
- Template Method pattern: ``generate`` / ``embed`` on concrete clients call
  ``_generate`` / ``_embed`` (abstract hooks) after running the middleware chain.
- Dependency Injection: the middleware pipeline and config are injected at
  construction time, keeping the base class testable without I/O.
- Open/Closed: new cross-cutting concerns are added as Middleware, not by
  editing this class.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from conduit_sdk.core.config import (
    ClientConfig,
)
from conduit_sdk.core.middleware import CallContext, Middleware, MiddlewarePipeline
from conduit_sdk.models.requests import AnyRequest
from conduit_sdk.models.responses import AnyResponse
from conduit_sdk.utils.cost import CostMiddleware
from conduit_sdk.utils.logging import LoggingMiddleware
from conduit_sdk.utils.rate_limit import RateLimitMiddleware
from conduit_sdk.utils.retry import RetryMiddleware


def _build_default_pipeline(config: ClientConfig) -> MiddlewarePipeline:
    """
    Factory that constructs the standard middleware stack from config.

    Order (outermost → innermost):
        Logging → RateLimit → Retry → Cost → [handler]

    Logging wraps everything so total latency (including retries) is captured.
    RateLimit gates before the retry loop to avoid hammering the bucket.
    Retry wraps Cost so aborted attempts don't inflate cost counters.
    """
    stack: list[Middleware] = []

    if config.logging.enabled:
        stack.append(LoggingMiddleware(config.logging))
    stack.append(RateLimitMiddleware(config.rate_limit))
    stack.append(RetryMiddleware(config.retry))
    if config.cost.enabled:
        stack.append(CostMiddleware(config.cost))

    return MiddlewarePipeline(stack)


class BaseClient(ABC):  # noqa: B024
    """
    Abstract base for all conduit_sdk clients.

    Subclasses should NOT override ``_execute`` directly.  Instead, implement
    the modality-specific abstract hook (``_generate``, ``_embed``, etc.) which
    is called by the concrete client subclass after the pipeline runs.

    Parameters
    ----------
    config:
        Immutable client configuration.
    middleware:
        Optional custom pipeline.  When omitted, a default pipeline is built
        from ``config``.  Pass an empty ``MiddlewarePipeline()`` to disable all
        middleware.

    Extension example
    -----------------
    ::

        class MyClient(LLMClient):
            async def _generate(self, request: LLMRequest) -> LLMResponse:
                # call your provider here
                raw = await my_provider_sdk.complete(request.messages)
                return LLMResponse(content=raw.text, usage=...)
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        middleware: MiddlewarePipeline | None = None,
    ) -> None:
        self._config: ClientConfig = config or ClientConfig()
        self._pipeline: MiddlewarePipeline = middleware or _build_default_pipeline(self._config)

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ClientConfig:
        return self._config

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def model(self) -> str:
        return self._config.model

    # ------------------------------------------------------------------
    # Internal pipeline execution (shared by all modalities)
    # ------------------------------------------------------------------

    async def _execute(
        self,
        request: AnyRequest,
        handler: Any,  # Callable[[CallContext], Awaitable[AnyResponse]]
    ) -> AnyResponse:
        """Run the middleware pipeline then invoke ``handler``."""
        ctx = CallContext(
            request=request,
            provider=self._config.provider,
            model=self._config.model,
        )
        return await self._pipeline.execute(ctx, handler)

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional override)
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: B027
        """Release any underlying HTTP sessions or resources.

        Override if your provider SDK requires explicit teardown.
        """

    async def __aenter__(self) -> BaseClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"provider={self._config.provider!r}, "
            f"model={self._config.model!r})"
        )
