"""
Groq provider — pip install llm-conduit[groq]

Ultra-fast LLM inference via Groq LPU hardware.
The API is OpenAI-compatible, so request/response shapes are identical.

    GroqLLMClient  → LLMClient  (llama-3.3-70b-versatile, llama-3.1-8b-instant, …)

Quick start::

    from conduit_sdk.providers.groq import GroqLLMClient
    from conduit_sdk.core.config import ClientConfig
    from conduit_sdk.models.requests import LLMRequest

    client = GroqLLMClient(ClientConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key="gsk_...",   # or set GROQ_API_KEY env var
    ))
    response = await client.generate(
        LLMRequest.Builder().user("What is RLHF?").max_tokens(200).build()
    )
    print(response.content)
"""

from conduit_sdk.providers.groq.llm import GroqLLMClient

__all__ = ["GroqLLMClient"]
