"""
Example 08 — Config-driven provider selection via registries
=============================================================
Shows:
  - Registering multiple providers in ModelRegistry and ProviderRegistry
  - Resolving a client at runtime from a string config (e.g. YAML / env var)
  - Switching providers without changing application code
  - Querying the registry for available models and providers
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from conduit_sdk.clients import EmbeddingClient, ImageGenClient, LLMClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.core.middleware import MiddlewarePipeline
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import EmbeddingRequest, ImageGenRequest, LLMRequest
from conduit_sdk.models.responses import (
    Embedding,
    EmbeddingResponse,
    FinishReason,
    GeneratedImage,
    ImageGenResponse,
    LLMResponse,
)
from conduit_sdk.registry import ModelDefinition, ModelRegistry, ProviderRegistry

# ---------------------------------------------------------------------------
# Mock provider implementations (one per provider)
# ---------------------------------------------------------------------------


def _bare() -> MiddlewarePipeline:
    return MiddlewarePipeline([])


class ProviderALLM(LLMClient):
    """Simulates Provider A's LLM."""

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(
                role=MessageRole.ASSISTANT, content=f"[Provider-A] {request.messages[-1].content}"
            ),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model=self.config.model,
            provider="provider-a",
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        yield "[Provider-A] "


class ProviderBLLM(LLMClient):
    """Simulates Provider B's LLM."""

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(
                role=MessageRole.ASSISTANT, content=f"[Provider-B] {request.messages[-1].content}"
            ),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=12, completion_tokens=25, total_tokens=37),
            model=self.config.model,
            provider="provider-b",
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        yield "[Provider-B] "


class ProviderAImageGen(ImageGenClient):
    async def _generate(self, request: ImageGenRequest) -> ImageGenResponse:
        return ImageGenResponse(
            images=[
                GeneratedImage(url=f"https://provider-a.example.com/{request.prompt[:20]}.png")
            ],
            usage=Usage(image_count=1),
            model=self.config.model,
            provider="provider-a",
        )


class ProviderAEmbedder(EmbeddingClient):
    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[
                Embedding(index=i, vector=[0.1, 0.2, 0.3]) for i in range(len(request.inputs))
            ],
            usage=Usage(prompt_tokens=10, total_tokens=10, embedding_count=len(request.inputs)),
            model=self.config.model,
            provider="provider-a",
        )


# ---------------------------------------------------------------------------
# 1. Register model definitions (metadata catalogue)
# ---------------------------------------------------------------------------

model_registry = ModelRegistry()  # isolated instance — not the global singleton

model_registry.register_many(
    [
        ModelDefinition(
            name="provider-a/llm-large",
            provider="provider-a",
            modality="llm",
            aliases=["pa-large"],
            context_window=128_000,
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.015,
            metadata={"supports_function_calling": True},
        ),
        ModelDefinition(
            name="provider-b/llm-fast",
            provider="provider-b",
            modality="llm",
            aliases=["pb-fast"],
            context_window=32_000,
            input_cost_per_1k_tokens=0.001,
            output_cost_per_1k_tokens=0.003,
        ),
        ModelDefinition(
            name="provider-a/image-hd",
            provider="provider-a",
            modality="image",
            aliases=["pa-image"],
            metadata={"image_cost_per_unit": 0.04},
        ),
        ModelDefinition(
            name="provider-a/embed-v2",
            provider="provider-a",
            modality="embedding",
            aliases=["pa-embed"],
            metadata={"embedding_cost_per_1k_tokens": 0.0001},
        ),
    ]
)


# ---------------------------------------------------------------------------
# 2. Register provider factories
# ---------------------------------------------------------------------------

provider_registry = ProviderRegistry()  # isolated instance

provider_registry.register_factory(
    "provider-a", "llm", lambda cfg: ProviderALLM(config=cfg, middleware=_bare())
).register_factory(
    "provider-b", "llm", lambda cfg: ProviderBLLM(config=cfg, middleware=_bare())
).register_factory(
    "provider-a", "image", lambda cfg: ProviderAImageGen(config=cfg, middleware=_bare())
).register_factory(
    "provider-a", "embedding", lambda cfg: ProviderAEmbedder(config=cfg, middleware=_bare())
)


# ---------------------------------------------------------------------------
# 3. Application code — provider-agnostic, driven by string config
# ---------------------------------------------------------------------------


def get_llm_client(model_alias: str) -> LLMClient:
    """
    Resolve a client purely from a model alias string.
    In production this alias comes from an env var or YAML config.
    """
    defn = model_registry.resolve(model_alias)
    config = ClientConfig(
        provider=defn.provider,
        model=defn.name,
        api_key=os.getenv(f"{defn.provider.upper()}_API_KEY", "mock-key"),
    )
    return provider_registry.create_client(defn.provider, "llm", config)  # type: ignore[return-value]


async def main() -> None:
    print("=" * 55)
    print("Example 08 — Registry-Driven Provider Selection")
    print("=" * 55)

    # --- Inspect the registry ---
    print("\n[Available LLM models]")
    for m in model_registry.list_models(modality="llm"):
        print(f"  {m.name}  aliases={m.aliases}  ctx={m.context_window}")

    print("\n[Registered provider factories]")
    for provider, modality in provider_registry.list_providers():
        print(f"  {provider} / {modality}")

    # --- Switch providers via a config string ---
    request = LLMRequest(messages=[Message.user("Hello, which provider are you?")])

    for alias in ["pa-large", "pb-fast"]:
        print(f"\n[Using model alias: {alias!r}]")
        client = get_llm_client(alias)
        response = await client.generate(request)
        print(f"  Provider : {response.provider}")
        print(f"  Response : {response.content}")
        print(f"  Tokens   : {response.usage.total_tokens}")

    # --- Resolve model metadata for pricing display ---
    print("\n[Model pricing lookup]")
    for alias in ["pa-large", "pb-fast", "pa-image", "pa-embed"]:
        defn = model_registry.resolve(alias)
        cost_in = (
            defn.input_cost_per_1k_tokens
            or defn.metadata.get("image_cost_per_unit")
            or defn.metadata.get("embedding_cost_per_1k_tokens")
            or "-"
        )
        cost_out = defn.output_cost_per_1k_tokens or "-"
        print(f"  {alias:<12} → {defn.name:<30} in=${cost_in}  out=${cost_out}")


if __name__ == "__main__":
    asyncio.run(main())
