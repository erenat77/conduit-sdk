"""
conduit_sdk — A generic, provider-agnostic SDK for AI model clients.

Quickstart
----------
Subclass any of the abstract clients to wire in your provider:

    from conduit_sdk.clients import LLMClient
    from conduit_sdk.models.requests import LLMRequest
    from conduit_sdk.models.responses import LLMResponse

    class MyOpenAIClient(LLMClient):
        async def _generate(self, request: LLMRequest) -> LLMResponse:
            ...  # call openai SDK here

    client = MyOpenAIClient(config=ClientConfig(model="gpt-4o"))
    response = await client.generate(LLMRequest(messages=[...]))
"""

from conduit_sdk.clients.embedding import EmbeddingClient
from conduit_sdk.clients.image import ImageGenClient
from conduit_sdk.clients.llm import LLMClient
from conduit_sdk.clients.video import VideoGenClient
from conduit_sdk.core.base import BaseClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.registry.model_registry import ModelRegistry
from conduit_sdk.registry.provider_registry import ProviderRegistry

__all__ = [
    "BaseClient",
    "ClientConfig",
    "LLMClient",
    "ImageGenClient",
    "VideoGenClient",
    "EmbeddingClient",
    "ModelRegistry",
    "ProviderRegistry",
]

__version__ = "0.1.0"
