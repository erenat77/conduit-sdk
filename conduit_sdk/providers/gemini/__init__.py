"""
Gemini provider — pip install llm-conduit[gemini]

Google Gemini models via the google-genai unified SDK.

    GeminiLLMClient       → LLMClient        (gemini-2.0-flash, gemini-1.5-pro, …)
    GeminiEmbeddingClient → EmbeddingClient  (text-embedding-004, …)

Quick start::

    from conduit_sdk.providers.gemini import GeminiLLMClient
    from conduit_sdk.core.config import ClientConfig
    from conduit_sdk.models.requests import LLMRequest

    client = GeminiLLMClient(ClientConfig(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="AIza...",   # or set GOOGLE_API_KEY env var
    ))
    response = await client.generate(
        LLMRequest.Builder().user("What is RLHF?").max_tokens(200).build()
    )
    print(response.content)
"""

from conduit_sdk.providers.gemini.embedding import GeminiEmbeddingClient
from conduit_sdk.providers.gemini.llm import GeminiLLMClient

__all__ = ["GeminiLLMClient", "GeminiEmbeddingClient"]
