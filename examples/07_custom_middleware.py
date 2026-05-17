"""
Example 07 — Writing custom middleware
========================================
Shows:
  - Subclassing Middleware and implementing __call__
  - Reading and mutating CallContext.metadata
  - Short-circuiting the chain (cache middleware)
  - Composing a fully custom pipeline
  - Three practical middleware examples:
      1. RequestIdMiddleware  — inject a trace/request ID
      2. CacheMiddleware      — in-memory response cache (short-circuit)
      3. LatencyBudgetMiddleware — raise if call exceeds a deadline
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator

from conduit_sdk.clients import LLMClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import CallContext, Middleware, MiddlewarePipeline, NextCall
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import AnyResponse, FinishReason, LLMResponse

# ---------------------------------------------------------------------------
# Middleware 1 — Inject a unique request ID into context metadata
# ---------------------------------------------------------------------------


class RequestIdMiddleware(Middleware):
    """
    Stamps every call with a UUID so downstream middleware and logs
    can correlate events for a single request.
    """

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        ctx.metadata["request_id"] = str(uuid.uuid4())
        print(f"  [RequestId] → {ctx.metadata['request_id']}")
        return await next_call(ctx)


# ---------------------------------------------------------------------------
# Middleware 2 — In-memory LRU-style cache (short-circuits the chain)
# ---------------------------------------------------------------------------


class CacheMiddleware(Middleware):
    """
    Caches responses keyed by a hash of the serialized request.
    On a cache hit, returns immediately without calling downstream middleware
    or the provider — demonstrating pipeline short-circuiting.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._cache: dict[str, AnyResponse] = {}
        self._max_size = max_size

    def _cache_key(self, ctx: CallContext) -> str:
        raw = ctx.request.model_dump_json(exclude={"extra"})
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        key = self._cache_key(ctx)

        if key in self._cache:
            ctx.metadata["cache_hit"] = True
            print(f"  [Cache] HIT  key={key}")
            return self._cache[key]

        ctx.metadata["cache_hit"] = False
        print(f"  [Cache] MISS key={key}")
        response = await next_call(ctx)

        if len(self._cache) < self._max_size:
            self._cache[key] = response
        return response

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Middleware 3 — Latency budget (deadline enforcement)
# ---------------------------------------------------------------------------


class LatencyBudgetMiddleware(Middleware):
    """
    Raises TimeoutError if the downstream call exceeds ``budget_ms``
    milliseconds. Useful for enforcing SLOs in production.
    """

    def __init__(self, budget_ms: float) -> None:
        self._budget = budget_ms / 1000.0

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        try:
            return await asyncio.wait_for(next_call(ctx), timeout=self._budget)
        except TimeoutError:
            from conduit_sdk.core.exceptions import TimeoutError as SDKTimeout

            elapsed = ctx.elapsed_seconds * 1000
            raise SDKTimeout(
                f"Call exceeded latency budget of {self._budget * 1000:.0f}ms "
                f"(elapsed {elapsed:.0f}ms)",
                provider=ctx.provider,
            ) from None


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class SlowMockLLM(LLMClient):
    def __init__(self, delay_ms: float = 0, **kwargs):
        super().__init__(**kwargs)
        self._delay = delay_ms / 1000.0

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        if self._delay:
            await asyncio.sleep(self._delay)
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content="Done."),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
            model=self.config.model,
            provider=self.config.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        yield "Done."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 55)
    print("Example 07 — Custom Middleware")
    print("=" * 55)

    cache = CacheMiddleware(max_size=64)

    pipeline = MiddlewarePipeline(
        [
            RequestIdMiddleware(),
            LatencyBudgetMiddleware(budget_ms=500),
            cache,
        ]
    )

    client = SlowMockLLM(
        delay_ms=10,
        config=ClientConfig(provider="mock", model="mock-v1"),
        middleware=pipeline,
    )

    request = LLMRequest(messages=[Message.user("What is 2 + 2?")])

    # First call — cache miss
    print("\n[Call 1 — cache miss]")
    resp1 = await client.generate(request)
    print(f"  Response : {resp1.content}")
    print(f"  Cache size: {cache.size}")

    # Second call (same request) — cache hit
    print("\n[Call 2 — cache hit]")
    resp2 = await client.generate(request)
    print(f"  Response : {resp2.content}")

    # Different request — cache miss
    print("\n[Call 3 — different request, cache miss]")
    resp3 = await client.generate(LLMRequest(messages=[Message.user("What is 3 + 3?")]))
    print(f"  Response : {resp3.content}")
    print(f"  Cache size: {cache.size}")

    # Trigger latency budget breach
    print("\n[Call 4 — latency budget exceeded]")
    slow_client = SlowMockLLM(
        delay_ms=600,  # exceeds 500ms budget
        config=ClientConfig(provider="mock", model="mock-slow"),
        middleware=MiddlewarePipeline([LatencyBudgetMiddleware(budget_ms=500)]),
    )
    try:
        await slow_client.generate(request)
    except Exception as e:
        print(f"  Caught: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
