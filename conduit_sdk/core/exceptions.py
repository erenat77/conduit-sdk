"""
Structured exception hierarchy for conduit_sdk.

Design
------
All SDK exceptions derive from ``ModelSDKError`` so callers can catch the
entire family with a single ``except ModelSDKError`` clause while still
being able to handle specific sub-types when needed.

    ModelSDKError
    ├── ConfigurationError      — bad client config / missing keys
    ├── ValidationError         — request schema violation
    ├── ProviderError           — upstream provider returned an error
    │   ├── RateLimitError      — 429 / quota exceeded
    │   ├── AuthenticationError — 401 / bad credentials
    │   └── TimeoutError        — request exceeded deadline
    ├── MiddlewareError         — error inside middleware pipeline
    └── RegistryError           — model / provider not found in registry
"""

from __future__ import annotations


class ModelSDKError(Exception):
    """Base class for all conduit_sdk exceptions."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.args[0]!r}, provider={self.provider!r})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(ModelSDKError):
    """Raised when a client is misconfigured (missing keys, invalid values)."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError(ModelSDKError):
    """Raised when a request fails schema validation before being sent."""


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------


class ProviderError(ModelSDKError):
    """Raised when the upstream provider returns a non-success response."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        raw_response: object = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.status_code = status_code
        self.raw_response = raw_response


class RateLimitError(ProviderError):
    """Raised when the provider enforces rate limits (HTTP 429 or quota exceeded)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        provider: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when credentials are rejected by the provider."""

    def __init__(
        self, message: str = "Authentication failed", *, provider: str | None = None
    ) -> None:
        super().__init__(message, provider=provider, status_code=401)


class TimeoutError(ProviderError):
    """Raised when a request exceeds its configured deadline."""

    def __init__(self, message: str = "Request timed out", *, provider: str | None = None) -> None:
        super().__init__(message, provider=provider)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class MiddlewareError(ModelSDKError):
    """Raised when a middleware component encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RegistryError(ModelSDKError):
    """Raised when a model or provider lookup in the registry fails."""
