"""
Built-in provider adapters.

Each provider lives in its own sub-package and is available as an optional
install extra so that users only pull in the SDKs they actually need.

    pip install llm-conduit[openai]

Then import:

    from conduit_sdk.providers.openai import (
        OpenAILLMClient, OpenAIEmbeddingClient, OpenAIImageClient
    )
"""
