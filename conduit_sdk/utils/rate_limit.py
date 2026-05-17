"""
RateLimitMiddleware — token-bucket rate limiter.

Design
------
- Token Bucket algorithm: allows short bursts up to ``burst`` tokens while
  enforcing a long-run rate of ``requests_per_minute``.
- The ``TokenBucketRateLimiter`` is decoupled from the middleware and can be
  used standalone (e.g. shared across multiple client instances).
- Uses ``asyncio.Lock`` so it is safe under concurrent coroutines.
"""

from __future__ import annotations

import asyncio
import time

from conduit_sdk.core.config import RateLimitConfig
from conduit_sdk.core.middleware import CallContext, Middleware, NextCall
from conduit_sdk.models.responses import AnyResponse


class TokenBucketRateLimiter:
    """
    Async token-bucket rate limiter.

    Parameters
    ----------
    rate:
        Steady-state fill rate in tokens per second.
    burst:
        Maximum token accumulation (burst capacity).
    """

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self._rate = rate
        self._burst = float(burst)
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                self._last_refill = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Sleep until we have enough tokens
                deficit = tokens - self._tokens
                sleep_for = deficit / self._rate
                await asyncio.sleep(sleep_for)

    @property
    def available_tokens(self) -> float:
        """Approximate number of tokens currently in the bucket."""
        return self._tokens


class RateLimitMiddleware(Middleware):
    """
    Middleware that gates requests through a ``TokenBucketRateLimiter``.

    Each instance owns its own limiter, so separate client instances have
    independent limits by default.  For shared limits, inject a pre-built
    ``TokenBucketRateLimiter`` into multiple middleware instances.

    Parameters
    ----------
    config:
        ``RateLimitConfig`` controlling the token bucket parameters.
    limiter:
        Optional pre-built limiter (e.g. for sharing across clients).
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        *,
        limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        cfg = config or RateLimitConfig()
        rate_per_second = cfg.requests_per_minute / 60.0
        self._limiter = limiter or TokenBucketRateLimiter(rate=rate_per_second, burst=cfg.burst)

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        await self._limiter.acquire()
        return await next_call(ctx)
