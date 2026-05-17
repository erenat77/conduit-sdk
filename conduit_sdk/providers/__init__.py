"""
Built-in provider adapters.

Each provider lives in its own sub-package and is available as an optional
install extra so that users only pull in the SDKs they actually need.

Install examples::

    pip install llm-conduit[openai]
    pip install llm-conduit[anthropic]
    pip install llm-conduit[groq]
    pip install llm-conduit[gemini]
    pip install llm-conduit[mistral]
    pip install llm-conduit[cohere]
    pip install llm-conduit[replicate]
    pip install llm-conduit[all]     # everything

Then import::

    from conduit_sdk.providers.openai import OpenAILLMClient
    from conduit_sdk.providers.groq import GroqLLMClient
    from conduit_sdk.providers.gemini import GeminiLLMClient, GeminiEmbeddingClient
    from conduit_sdk.providers.mistral import MistralLLMClient, MistralEmbeddingClient
    from conduit_sdk.providers.cohere import CohereLLMClient, CohereEmbeddingClient
    from conduit_sdk.providers.replicate import ReplicateImageClient, ReplicateVideoClient
"""
