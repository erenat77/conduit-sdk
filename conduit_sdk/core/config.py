"""
ClientConfig — immutable configuration value object shared by all clients.

Design: Value Object pattern (Pydantic frozen model).  Passed into every
client at construction time; clients must not mutate it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetryConfig(BaseModel):
    """Retry policy configuration."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1, le=10)
    min_wait_seconds: float = Field(default=1.0, ge=0.0)
    max_wait_seconds: float = Field(default=60.0, ge=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    reraise: bool = True  # re-raise the last exception after all retries exhausted

    @model_validator(mode="after")
    def _validate_wait_bounds(self) -> RetryConfig:
        if self.min_wait_seconds > self.max_wait_seconds:
            raise ValueError("min_wait_seconds must be <= max_wait_seconds")
        return self


class RateLimitConfig(BaseModel):
    """Token-bucket rate limiter configuration."""

    model_config = ConfigDict(frozen=True)

    requests_per_minute: float = Field(default=60.0, gt=0)
    burst: int = Field(default=10, ge=1)  # max tokens above steady-state


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    level: str = "INFO"
    include_request_body: bool = False  # may contain PII — off by default
    include_response_body: bool = False


class CostConfig(BaseModel):
    """Cost tracking configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    # Optional: price per 1k tokens {input, output} — providers set these
    input_cost_per_1k_tokens: float | None = None
    output_cost_per_1k_tokens: float | None = None
    image_cost_per_unit: float | None = None
    video_cost_per_second: float | None = None
    embedding_cost_per_1k_tokens: float | None = None


class ClientConfig(BaseModel):
    """
    Immutable configuration value object for any model client.

    Pass a ``ClientConfig`` when constructing a client subclass:

        config = ClientConfig(
            model="gpt-4o",
            api_key="sk-...",
            timeout_seconds=30.0,
            retry=RetryConfig(max_attempts=5),
        )
        client = MyLLMClient(config=config)

    All fields are optional; sensible defaults are provided.  Provider-specific
    extras can be stored in ``extra``.
    """

    model_config = ConfigDict(frozen=True)

    # Core identity
    model: str = Field(default="", description="Model identifier string (e.g. 'gpt-4o')")
    provider: str = Field(default="", description="Provider name (e.g. 'openai', 'anthropic')")
    api_key: str | None = Field(default=None, repr=False)
    api_base_url: str | None = None

    # Request behaviour
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)  # convenience alias; used by RetryConfig if set

    # Sub-configs
    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cost: CostConfig = Field(default_factory=CostConfig)

    # Escape hatch for provider-specific settings
    extra: dict[str, Any] = Field(default_factory=dict)
