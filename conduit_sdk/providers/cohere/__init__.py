"""
Cohere provider — pip install llm-conduit[cohere]

Cohere models via the official cohere Python SDK (v2 client).

    CohereLLMClient       → LLMClient        (command-r-plus, command-r, …)
    CohereEmbeddingClient → EmbeddingClient  (embed-v4.0, embed-english-v3.0, …)

Quick start::

    from conduit_sdk.providers.cohere import CohereLLMClient
    from conduit_sdk.core.config import ClientConfig
    from conduit_sdk.models.requests import LLMRequest

    client = CohereLLMClient(ClientConfig(
        provider="cohere",
        model="command-r-plus-08-2024",
        api_key="...",   # or set COHERE_API_KEY env var
    ))
    response = await client.generate(
        LLMRequest.Builder().user("What is RAG?").max_tokens(200).build()
    )
    print(response.content)
"""

from conduit_sdk.providers.cohere.embedding import CohereEmbeddingClient
from conduit_sdk.providers.cohere.llm import CohereLLMClient

__all__ = ["CohereLLMClient", "CohereEmbeddingClient"]
