# Contributing to llm-conduit

Thank you for your interest in contributing. This guide covers everything you need to get started.

## Development setup

```bash
git clone https://github.com/your-org/llm-conduit
cd llm-conduit

# Install uv (fast Python package manager)
pip install uv

# Install project with dev dependencies
uv pip install --system -e ".[dev]"
```

## Running checks locally

```bash
# Tests
pytest tests/ -v

# Tests with coverage (must stay above 80%)
pytest tests/ --cov=conduit_sdk --cov-report=term-missing

# Lint
ruff check conduit_sdk tests examples

# Format
ruff format conduit_sdk tests examples

# Type check
mypy conduit_sdk
```

All four must pass before opening a PR. The CI will enforce them automatically.

## Where to put new code

The single most important rule: **new cross-cutting concerns go in middleware, not in client classes.**

| What you're adding | Where it goes |
|---|---|
| New modality (e.g. audio) | `conduit_sdk/models/requests.py`, `conduit_sdk/models/responses.py`, `conduit_sdk/clients/audio.py`, `conduit_sdk/core/protocols.py` |
| New middleware (e.g. tracing) | `conduit_sdk/utils/tracing.py`, add to `MiddlewarePipeline` default in `core/base.py` |
| New provider adapter | `examples/providers/myprovider_client.py` |
| New request/response field | `conduit_sdk/models/requests.py` or `responses.py` — keep `extra: dict` for provider-specific fields |
| New registry feature | `conduit_sdk/registry/` |

## Adding a provider adapter

1. Create `examples/providers/{provider}_client.py`
2. Subclass the relevant abstract client(s)
3. Implement the abstract method(s) — `_generate`, `_stream`, or `_embed`
4. Add a `ModelDefinition` example showing pricing and aliases
5. Open a PR — no test requirement for example adapters, but include a docstring showing usage

## Adding a new modality

This is a bigger change. Follow this checklist:

- [ ] Add `XxxRequest` and `XxxResponse` Pydantic models in `models/`
- [ ] Add `XxxClient` ABC in `clients/` with clear docstring on what to override
- [ ] Add `XxxProtocol` to `core/protocols.py`
- [ ] Add the new types to the `AnyRequest` / `AnyResponse` unions in `models/`
- [ ] Export from `conduit_sdk/__init__.py`
- [ ] Write tests in `tests/test_xxx.py` using a mock provider
- [ ] Add an example in `examples/`
- [ ] Update `README.md`

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add AudioGenClient modality
fix: retry on 503 ProviderError
docs: add Anthropic provider skeleton
chore: bump pydantic to 2.8
test: add streaming edge case for empty chunks
refactor: extract CostCalculator into standalone class
```

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Patch** (`0.1.x`): bug fixes, documentation, non-breaking improvements
- **Minor** (`0.x.0`): new features that are backwards-compatible
- **Major** (`x.0.0`): breaking changes to `BaseClient`, abstract client interfaces, or Pydantic model fields

Breaking changes to any abstract method signature or frozen Pydantic model **must** increment the major version and be documented in `CHANGELOG.md`.

## Opening a PR

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes and run the checks above
3. Open a PR against `main` using the PR template
4. A maintainer will review within 48 hours

## Code of conduct

Be direct, be respectful, focus on the code. We welcome contributors of all experience levels.
