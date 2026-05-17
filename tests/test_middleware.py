"""Tests for middleware pipeline, retry, rate-limit, cost, and logging."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from conduit_sdk.core.config import (
    CostConfig,
    LoggingConfig,
    RateLimitConfig,
    RetryConfig,
)
from conduit_sdk.core.exceptions import ProviderError, RateLimitError
from conduit_sdk.core.middleware import CallContext, MiddlewarePipeline
from conduit_sdk.models.common import Message, Usage
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason, LLMResponse
from conduit_sdk.utils.cost import CostCalculator, CostMiddleware, CostTracker
from conduit_sdk.utils.logging import LoggingMiddleware, StructuredLogger
from conduit_sdk.utils.rate_limit import RateLimitMiddleware, TokenBucketRateLimiter
from conduit_sdk.utils.retry import RetryMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_request() -> LLMRequest:
    return LLMRequest(messages=[Message.user("test")])


def _llm_response(**kwargs) -> LLMResponse:
    return LLMResponse(
        message=Message.assistant("ok"),
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        **kwargs,
    )


async def _ok_handler(ctx: CallContext) -> LLMResponse:
    return _llm_response()


# ---------------------------------------------------------------------------
# MiddlewarePipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_pipeline_calls_handler():
    pipeline = MiddlewarePipeline([])
    ctx = CallContext(request=_llm_request())
    resp = await pipeline.execute(ctx, _ok_handler)
    assert resp.message.content == "ok"


@pytest.mark.asyncio
async def test_middleware_order():
    """Middleware fires in left-to-right order on the way in."""
    order: list[str] = []

    class TrackMiddleware:
        def __init__(self, name: str):
            self.name = name

        async def __call__(self, ctx, next_call):
            order.append(f"pre:{self.name}")
            resp = await next_call(ctx)
            order.append(f"post:{self.name}")
            return resp

    from conduit_sdk.core.middleware import Middleware

    class A(Middleware):
        async def __call__(self, ctx, next_call):
            order.append("pre:A")
            resp = await next_call(ctx)
            order.append("post:A")
            return resp

    class B(Middleware):
        async def __call__(self, ctx, next_call):
            order.append("pre:B")
            resp = await next_call(ctx)
            order.append("post:B")
            return resp

    pipeline = MiddlewarePipeline([A(), B()])
    ctx = CallContext(request=_llm_request())
    await pipeline.execute(ctx, _ok_handler)

    assert order == ["pre:A", "pre:B", "post:B", "post:A"]


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    attempts = 0

    async def flaky_handler(ctx: CallContext) -> LLMResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError()
        return _llm_response()

    mw = RetryMiddleware(RetryConfig(max_attempts=3, min_wait_seconds=0, max_wait_seconds=0))
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())
    resp = await pipeline.execute(ctx, flaky_handler)
    assert attempts == 2
    assert resp.message.content == "ok"


@pytest.mark.asyncio
async def test_retry_reraises_after_exhaustion():
    async def always_fail(ctx):
        raise RateLimitError("quota")

    mw = RetryMiddleware(RetryConfig(max_attempts=2, min_wait_seconds=0, max_wait_seconds=0))
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())
    with pytest.raises(RateLimitError):
        await pipeline.execute(ctx, always_fail)


@pytest.mark.asyncio
async def test_non_retryable_error_not_retried():
    attempts = 0

    async def fail_once(ctx):
        nonlocal attempts
        attempts += 1
        raise ValueError("not retryable")

    mw = RetryMiddleware(RetryConfig(max_attempts=3, min_wait_seconds=0, max_wait_seconds=0))
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())
    with pytest.raises(ValueError):
        await pipeline.execute(ctx, fail_once)
    assert attempts == 1


# ---------------------------------------------------------------------------
# RateLimitMiddleware / TokenBucket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_bucket_allows_burst():
    bucket = TokenBucketRateLimiter(rate=10.0, burst=5)
    for _ in range(5):
        await bucket.acquire()  # should not block


@pytest.mark.asyncio
async def test_rate_limit_middleware_passes_through():
    mw = RateLimitMiddleware(RateLimitConfig(requests_per_minute=600, burst=100))
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())
    resp = await pipeline.execute(ctx, _ok_handler)
    assert resp.message.content == "ok"


# ---------------------------------------------------------------------------
# CostMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_middleware_attaches_cost():
    config = CostConfig(
        enabled=True,
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03,
    )
    mw = CostMiddleware(config)
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())
    resp = await pipeline.execute(ctx, _ok_handler)
    assert resp.cost is not None
    assert resp.cost.total_cost > 0


@pytest.mark.asyncio
async def test_cost_tracker_accumulates():
    config = CostConfig(
        enabled=True,
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03,
    )
    tracker = CostTracker()
    mw = CostMiddleware(config, tracker=tracker)
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())

    await pipeline.execute(ctx, _ok_handler)
    await pipeline.execute(ctx, _ok_handler)

    assert tracker.call_count == 2
    assert tracker.total_usage.total_tokens == 20
    assert tracker.total_cost.total_cost > 0


def test_cost_calculator_zero_without_pricing():
    calc = CostCalculator(CostConfig())
    cost = calc.calculate(Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000))
    assert cost.total_cost == 0.0


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logging_middleware_fires_events():
    mock_logger = MagicMock(spec=StructuredLogger)
    mw = LoggingMiddleware(LoggingConfig(), logger=mock_logger)
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())
    await pipeline.execute(ctx, _ok_handler)

    mock_logger.request_start.assert_called_once()
    mock_logger.request_end.assert_called_once()
    mock_logger.request_error.assert_not_called()


@pytest.mark.asyncio
async def test_logging_middleware_fires_error_on_exception():
    mock_logger = MagicMock(spec=StructuredLogger)
    mw = LoggingMiddleware(LoggingConfig(), logger=mock_logger)
    pipeline = MiddlewarePipeline([mw])
    ctx = CallContext(request=_llm_request())

    async def fail(ctx):
        raise ProviderError("boom", provider="test")

    with pytest.raises(ProviderError):
        await pipeline.execute(ctx, fail)

    mock_logger.request_error.assert_called_once()
