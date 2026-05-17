"""
OpenAI provider — pip install conduit-sdk[openai]

Exposes three clients that implement the conduit_sdk abstract interfaces:

    OpenAILLMClient      → LLMClient     (gpt-4o, gpt-4-turbo, o1-*, …)
    OpenAIEmbeddingClient → EmbeddingClient (text-embedding-3-*)
    OpenAIImageClient    → ImageGenClient  (dall-e-3, dall-e-2)

Quick start::

    from conduit_sdk.providers.openai import OpenAILLMClient
    from conduit_sdk.core.config import ClientConfig, CostConfig
    from conduit_sdk.models.common import Message
    from conduit_sdk.models.requests import LLMRequest

    client = OpenAILLMClient(ClientConfig(
        model="gpt-4o",
        api_key="sk-...",          # or set OPENAI_API_KEY env var
        cost=CostConfig(
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.015,
        ),
    ))
    response = await client.generate(LLMRequest(messages=[Message.user("Hello!")]))
    print(response.content)
"""

from conduit_sdk.providers.openai.embedding import OpenAIEmbeddingClient
from conduit_sdk.providers.openai.image import OpenAIImageClient
from conduit_sdk.providers.openai.llm import OpenAILLMClient

__all__ = ["OpenAILLMClient", "OpenAIEmbeddingClient", "OpenAIImageClient"]
