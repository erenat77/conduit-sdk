"""
RetryMiddleware — exponential back-off retry via tenacity.

Design: Strategy pattern — the retry policy (``RetryConfig``) is a value
object injected at construction time, keeping policy decisions separate from
mechanism.
"""

from __future__ import annotations

import logging

import tenacity

from conduit_sdk.core.config import RetryConfig
from conduit_sdk.core.exceptions import RateLimitError
from conduit_sdk.core.middleware import CallContext, Middleware, NextCall
from conduit_sdk.models.responses import AnyResponse

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that should be retried."""
    from conduit_sdk.core.exceptions import ProviderError, TimeoutError

    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    # Generic provider errors with 5xx codes
    if isinstance(exc, ProviderError) and exc.status_code is not None:
        return exc.status_code >= 500
    return False


class RetryMiddleware(Middleware):
    """
    Retries the downstream chain on transient errors using exponential back-off.

    Parameters
    ----------
    config:
        ``RetryConfig`` value object controlling attempt count and wait bounds.
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        self._config = config or RetryConfig()

    def _build_retryer(self) -> tenacity.AsyncRetrying:
        cfg = self._config
        return tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception(_is_retryable),
            stop=tenacity.stop_after_attempt(cfg.max_attempts),
            wait=tenacity.wait_exponential(
                multiplier=cfg.multiplier,
                min=cfg.min_wait_seconds,
                max=cfg.max_wait_seconds,
            ),
            reraise=cfg.reraise,
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        )

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        retryer = self._build_retryer()
        async for attempt in retryer:
            with attempt:
                return await next_call(ctx)
        # tenacity reraises the last exception; this line is unreachable
        raise RuntimeError("Retry loop exited unexpectedly")  # pragma: no cover
