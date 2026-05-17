"""
LoggingMiddleware — structured pre/post-call logging.

Design: Observer pattern — the middleware fires structured log events without
coupling the client to any specific log sink.  Users can attach standard
``logging.Handler`` subclasses (JSON, CloudWatch, Datadog, etc.) externally.
"""

from __future__ import annotations

import logging
from typing import Any

from conduit_sdk.core.config import LoggingConfig
from conduit_sdk.core.middleware import CallContext, Middleware, NextCall
from conduit_sdk.models.responses import AnyResponse

_SDK_LOGGER = logging.getLogger("conduit_sdk")


class StructuredLogger:
    """
    Thin wrapper around Python's ``logging`` that emits structured dicts as
    the ``extra`` payload, making it easy to parse with log aggregators.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or _SDK_LOGGER

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        self._log.log(level, event, extra={"sdk_event": event, **fields})

    def request_start(self, ctx: CallContext, include_body: bool = False) -> None:
        payload: dict[str, Any] = {
            "provider": ctx.provider,
            "model": ctx.model,
            "request_type": type(ctx.request).__name__,
        }
        if include_body:
            payload["request"] = ctx.request.model_dump()
        self._emit(logging.INFO, "conduit_sdk.request.start", **payload)

    def request_end(
        self,
        ctx: CallContext,
        response: AnyResponse,
        include_body: bool = False,
    ) -> None:
        payload: dict[str, Any] = {
            "provider": ctx.provider,
            "model": ctx.model,
            "latency_ms": round(ctx.elapsed_seconds * 1000, 2),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.cost is not None:
            payload["cost_usd"] = response.cost.total_cost
        if include_body:
            payload["response"] = response.model_dump(exclude={"raw_response"})
        self._emit(logging.INFO, "conduit_sdk.request.end", **payload)

    def request_error(self, ctx: CallContext, exc: BaseException) -> None:
        self._emit(
            logging.ERROR,
            "conduit_sdk.request.error",
            provider=ctx.provider,
            model=ctx.model,
            latency_ms=round(ctx.elapsed_seconds * 1000, 2),
            error_type=type(exc).__name__,
            error=str(exc),
        )


class LoggingMiddleware(Middleware):
    """
    Emits structured log events before and after every model call.

    Parameters
    ----------
    config:
        ``LoggingConfig`` controlling verbosity and PII redaction.
    logger:
        Optional custom ``StructuredLogger`` (for dependency injection in tests).
    """

    def __init__(
        self,
        config: LoggingConfig | None = None,
        *,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._config = config or LoggingConfig()
        self._logger = logger or StructuredLogger()

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        self._logger.request_start(ctx, include_body=self._config.include_request_body)
        try:
            response = await next_call(ctx)
            # Streaming responses are async generators — skip structured end-logging
            # (no usage or cost data available until the stream is fully consumed).
            if hasattr(response, "usage"):
                self._logger.request_end(
                    ctx, response, include_body=self._config.include_response_body
                )
            return response
        except Exception as exc:
            self._logger.request_error(ctx, exc)
            raise
