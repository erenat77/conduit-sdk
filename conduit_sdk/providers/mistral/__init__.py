"""
Mistral provider — pip install llm-conduit[mistral]

Mistral AI models via the official mistralai Python SDK.

    MistralLLMClient       → LLMClient        (mistral-large-latest, …)
    MistralEmbeddingClient → EmbeddingClient  (mistral-embed)

Quick start::

    from conduit_sdk.providers.mistral import MistralLLMClient
    from conduit_sdk.core.config import ClientConfig
    from conduit_sdk.models.requests import LLMRequest

    client = MistralLLMClient(ClientConfig(
        provider="mistral",
        model="mistral-large-latest",
        api_key="...",   # or set MISTRAL_API_KEY env var
    ))
    response = await client.generate(
        LLMRequest.Builder().user("What is mixture of experts?").max_tokens(200).build()
    )
    print(response.content)
"""

from conduit_sdk.providers.mistral.embedding import MistralEmbeddingClient
from conduit_sdk.providers.mistral.llm import MistralLLMClient

__all__ = ["MistralLLMClient", "MistralEmbeddingClient"]
