"""Tests for ModelRegistry and ProviderRegistry."""

from __future__ import annotations

import pytest

from conduit_sdk.core.exceptions import RegistryError
from conduit_sdk.registry.model_registry import ModelDefinition, ModelRegistry
from conduit_sdk.registry.provider_registry import ProviderRegistry

# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ModelRegistry:
    """Isolated registry instance (not the global one)."""
    return ModelRegistry()


@pytest.fixture
def sample_definition() -> ModelDefinition:
    return ModelDefinition(
        name="openai/gpt-4o",
        provider="openai",
        modality="llm",
        aliases=["gpt-4o"],
        context_window=128_000,
        input_cost_per_1k_tokens=0.005,
        output_cost_per_1k_tokens=0.015,
    )


def test_register_and_resolve_by_name(registry, sample_definition):
    registry.register(sample_definition)
    defn = registry.resolve("openai/gpt-4o")
    assert defn.provider == "openai"


def test_resolve_by_alias(registry, sample_definition):
    registry.register(sample_definition)
    defn = registry.resolve("gpt-4o")
    assert defn.name == "openai/gpt-4o"


def test_resolve_unknown_raises(registry):
    with pytest.raises(RegistryError, match="not found"):
        registry.resolve("unknown/model")


def test_get_returns_none_for_missing(registry):
    assert registry.get("missing") is None


def test_list_models_empty(registry):
    assert registry.list_models() == []


def test_list_models_filtered_by_modality(registry):
    registry.register(ModelDefinition(name="p/llm-a", provider="p", modality="llm"))
    registry.register(ModelDefinition(name="p/img-a", provider="p", modality="image"))
    llms = registry.list_models(modality="llm")
    assert len(llms) == 1
    assert llms[0].modality == "llm"


def test_register_many(registry):
    defs = [ModelDefinition(name=f"p/model-{i}", provider="p", modality="llm") for i in range(3)]
    registry.register_many(defs)
    assert len(registry) == 3


def test_contains(registry, sample_definition):
    registry.register(sample_definition)
    assert "gpt-4o" in registry
    assert "unknown" not in registry


def test_overwrite_definition(registry, sample_definition):
    registry.register(sample_definition)
    updated = ModelDefinition(
        name="openai/gpt-4o",
        provider="openai",
        modality="llm",
        context_window=256_000,
    )
    registry.register(updated)
    assert registry.resolve("openai/gpt-4o").context_window == 256_000


# ---------------------------------------------------------------------------
# ProviderRegistry
# ---------------------------------------------------------------------------


@pytest.fixture
def provider_registry() -> ProviderRegistry:
    return ProviderRegistry()


def test_register_and_create_client(provider_registry, base_config):
    from tests.conftest import MockLLMClient

    provider_registry.register_factory(
        provider="mock",
        modality="llm",
        factory=lambda cfg: MockLLMClient(config=cfg, middleware=None),
    )
    client = provider_registry.create_client("mock", "llm", base_config)
    assert isinstance(client, MockLLMClient)


def test_create_client_unknown_raises(provider_registry):
    with pytest.raises(RegistryError, match="No factory"):
        provider_registry.create_client("ghost", "llm")


def test_list_providers(provider_registry):
    provider_registry.register_factory("openai", "llm", lambda cfg: object())
    provider_registry.register_factory("openai", "image", lambda cfg: object())
    provider_registry.register_factory("replicate", "video", lambda cfg: object())

    all_providers = provider_registry.list_providers()
    assert ("openai", "llm") in all_providers
    assert ("replicate", "video") in all_providers


def test_list_providers_filtered(provider_registry):
    provider_registry.register_factory("openai", "llm", lambda cfg: object())
    provider_registry.register_factory("openai", "image", lambda cfg: object())

    image_providers = provider_registry.list_providers(modality="image")
    assert all(m == "image" for _, m in image_providers)


def test_provider_contains(provider_registry):
    provider_registry.register_factory("openai", "llm", lambda cfg: object())
    assert ("openai", "llm") in provider_registry
    assert ("openai", "image") not in provider_registry
