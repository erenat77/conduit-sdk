"""
CostMiddleware — per-call cost estimation and session accumulation.

Design
------
- ``CostCalculator`` is a pure function object (Strategy) — swap in a
  different implementation for custom pricing models.
- ``CostTracker`` is a thread-safe accumulator that can be shared across
  multiple client instances in a session.
- ``CostMiddleware`` wires them together inside the pipeline.
"""

from __future__ import annotations

import threading

from conduit_sdk.core.config import CostConfig
from conduit_sdk.core.middleware import CallContext, Middleware, NextCall
from conduit_sdk.models.common import Cost, Usage
from conduit_sdk.models.responses import AnyResponse

# ---------------------------------------------------------------------------
# Calculator (Strategy)
# ---------------------------------------------------------------------------


class CostCalculator:
    """
    Estimates the monetary cost of a call from usage stats and pricing config.

    Override or replace to implement custom pricing logic.
    """

    def __init__(self, config: CostConfig) -> None:
        self._cfg = config

    def calculate(self, usage: Usage) -> Cost:
        cfg = self._cfg
        input_cost = 0.0
        output_cost = 0.0

        if cfg.input_cost_per_1k_tokens and usage.prompt_tokens:
            input_cost += (usage.prompt_tokens / 1000.0) * cfg.input_cost_per_1k_tokens

        if cfg.output_cost_per_1k_tokens and usage.completion_tokens:
            output_cost += (usage.completion_tokens / 1000.0) * cfg.output_cost_per_1k_tokens

        if cfg.embedding_cost_per_1k_tokens and usage.prompt_tokens and usage.embedding_count:
            input_cost += (usage.prompt_tokens / 1000.0) * cfg.embedding_cost_per_1k_tokens

        if cfg.image_cost_per_unit and usage.image_count:
            output_cost += usage.image_count * cfg.image_cost_per_unit

        if cfg.video_cost_per_second and usage.video_seconds:
            output_cost += usage.video_seconds * cfg.video_cost_per_second

        total = input_cost + output_cost
        return Cost(input_cost=input_cost, output_cost=output_cost, total_cost=total)


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class CostTracker:
    """
    Thread-safe session-level cost and usage accumulator.

    Use as a shared object when you want to aggregate across multiple calls:

        tracker = CostTracker()
        client = MyLLMClient(config=..., cost_tracker=tracker)
        ...
        print(f"Total cost: ${tracker.total_cost:.4f}")
        print(f"Total tokens: {tracker.total_usage.total_tokens}")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_cost = Cost()
        self._total_usage = Usage()
        self._call_count = 0

    def record(self, usage: Usage, cost: Cost) -> None:
        with self._lock:
            self._total_cost = self._total_cost + cost
            self._total_usage = self._total_usage + usage
            self._call_count += 1

    @property
    def total_cost(self) -> Cost:
        with self._lock:
            return self._total_cost

    @property
    def total_usage(self) -> Usage:
        with self._lock:
            return self._total_usage

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def reset(self) -> None:
        with self._lock:
            self._total_cost = Cost()
            self._total_usage = Usage()
            self._call_count = 0

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "call_count": self._call_count,
                "total_tokens": self._total_usage.total_tokens,
                "total_cost_usd": round(self._total_cost.total_cost, 6),
            }


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class CostMiddleware(Middleware):
    """
    Calculates estimated cost after each call and attaches it to the response.

    Also records into a ``CostTracker`` if one is provided, enabling
    session-level aggregation.

    Parameters
    ----------
    config:
        ``CostConfig`` with per-unit pricing.
    tracker:
        Optional shared ``CostTracker`` for session aggregation.
    calculator:
        Optional custom ``CostCalculator`` (default: uses ``config`` prices).
    """

    def __init__(
        self,
        config: CostConfig | None = None,
        *,
        tracker: CostTracker | None = None,
        calculator: CostCalculator | None = None,
    ) -> None:
        self._config = config or CostConfig()
        self._tracker = tracker
        self._calculator = calculator or CostCalculator(self._config)

    async def __call__(self, ctx: CallContext, next_call: NextCall) -> AnyResponse:
        response = await next_call(ctx)

        # Streaming responses are async generators — no usage data available,
        # so pass them through unchanged without attempting cost calculation.
        if not hasattr(response, "usage"):
            return response  # type: ignore[return-value]

        cost = self._calculator.calculate(response.usage)

        # Pydantic frozen model — rebuild with cost attached
        response = response.model_copy(update={"cost": cost})

        if self._tracker:
            self._tracker.record(response.usage, cost)

        return response
