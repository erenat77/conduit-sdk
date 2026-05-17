"""
Middleware pipeline — Chain of Responsibility pattern.

Design
------
Each ``Middleware`` receives a ``CallContext`` (the live request + metadata)
and a ``next_call`` callable that invokes the remainder of the chain.  This
mirrors the WSGI/ASGI middleware contract and makes it trivial to add, remove,
or reorder cross-cutting concerns without touching client logic.

    pipeline = MiddlewarePipeline([
        LoggingMiddleware(config.logging),
        RateLimitMiddleware(config.rate_limit),
        RetryMiddleware(config.retry),
        CostMiddleware(config.cost),
    ])

    response = await pipeline.execute(context, handler)
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from conduit_sdk.models.requests import AnyRequest
from conduit_sdk.models.responses import AnyResponse

# ---------------------------------------------------------------------------
# Context object — flows through the entire pipeline
# ---------------------------------------------------------------------------


@dataclass
class CallContext:
    """Mutable bag of state shared across all middleware for a single call."""

    request: AnyRequest
    provider: str = ""
    model: str = ""
    started_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at


# ---------------------------------------------------------------------------
# Middleware ABC
# ---------------------------------------------------------------------------

NextCall = Callable[[CallContext], Awaitable[AnyResponse]]


class Middleware:
    """
    Abstract base for a single middleware step.

    Subclass and implement ``__call__``.  Always call ``await next_call(ctx)``
    to continue the chain unless you intend to short-circuit it.

    Example::

        class TimingMiddleware(Middleware):
            async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
                t0 = time.monotonic()
                response = await next_call(ctx)
                print(f"Elapsed: {time.monotonic() - t0:.3f}s")
                return response
    """

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pipeline — composes middleware in order
# ---------------------------------------------------------------------------


class MiddlewarePipeline:
    """
    Composes a list of ``Middleware`` instances into a single callable chain.

    Middleware is applied left-to-right: the first item wraps all subsequent
    ones, so it runs first on the way *in* and last on the way *out*.

    Parameters
    ----------
    middleware:
        Ordered list of middleware to apply.
    """

    def __init__(self, middleware: list[Middleware] | None = None) -> None:
        self._stack: list[Middleware] = list(middleware or [])

    def add(self, mw: Middleware) -> MiddlewarePipeline:
        """Append a middleware to the end of the stack (fluent API)."""
        self._stack.append(mw)
        return self

    async def execute(
        self,
        ctx: CallContext,
        handler: Callable[[CallContext], Awaitable[AnyResponse]],
    ) -> AnyResponse:
        """
        Run the pipeline, terminating with ``handler`` (the actual provider call).

        Parameters
        ----------
        ctx:
            Shared call context.
        handler:
            The innermost callable that performs the real model request.
        """

        def build_chain(index: int) -> NextCall:
            if index >= len(self._stack):
                return handler
            mw = self._stack[index]
            inner = build_chain(index + 1)
            return lambda c: mw(c, inner)

        chain = build_chain(0)
        return await chain(ctx)
