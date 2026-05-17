# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Nothing yet — contributions welcome!

---

## [0.1.0] — 2026-05-16

Initial release.

### Added
- `LLMClient` abstract base class with `generate()`, `stream()`, and `generate_sync()` 
- `ImageGenClient` abstract base class with `generate()` and `generate_sync()`
- `VideoGenClient` abstract base class with `generate()` and `generate_sync()`
- `EmbeddingClient` abstract base class with `embed()` and `embed_sync()`
- `BaseClient` with async context manager support and middleware injection
- Pydantic v2 request models: `LLMRequest`, `ImageGenRequest`, `VideoGenRequest`, `EmbeddingRequest`
- Pydantic v2 response models: `LLMResponse`, `ImageGenResponse`, `VideoGenResponse`, `EmbeddingResponse`
- Shared value objects: `Message`, `MessageRole`, `Usage`, `Cost`
- `MiddlewarePipeline` — composable Chain of Responsibility pipeline
- `RetryMiddleware` — exponential back-off via tenacity; retries on `RateLimitError`, `TimeoutError`, 5xx
- `RateLimitMiddleware` — token-bucket rate limiter with configurable burst
- `CostMiddleware` + `CostTracker` — per-call cost estimation and session accumulation
- `LoggingMiddleware` + `StructuredLogger` — structured pre/post-call log events
- `ClientConfig` — immutable value object with `RetryConfig`, `RateLimitConfig`, `LoggingConfig`, `CostConfig`
- `ModelRegistry` — thread-safe catalogue of `ModelDefinition`s with alias resolution
- `ProviderRegistry` — Abstract Factory mapping `(provider, modality)` → client factory
- Structural `Protocol` definitions for all four modalities (PEP 544, `runtime_checkable`)
- Structured exception hierarchy: `ModelSDKError`, `ProviderError`, `RateLimitError`, `AuthenticationError`, `TimeoutError`, `ConfigurationError`, `ValidationError`, `RegistryError`
- 52-test suite covering all modalities, middleware, registry, and protocols (mock providers only — no API keys)
- 9 standalone examples with mock providers
- Provider adapter skeletons: OpenAI (LLM + Embedding), Anthropic (LLM), Replicate (Image), Runway (Video)
- Architecture SVG diagram (`docs/architecture.svg`)
- GitHub Actions: CI (lint + typecheck + test matrix × 3.10/3.11/3.12 + examples smoke test), publish (PyPI + TestPyPI via OIDC), CodeQL security scan
- `CONTRIBUTING.md`, PR template, bug/feature/provider issue templates

[Unreleased]: https://github.com/your-org/conduit-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/conduit-sdk/releases/tag/v0.1.0
