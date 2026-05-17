"""
Anthropic provider — pip install conduit-sdk[anthropic]

Exposes one client:

    AnthropicLLMClient  → LLMClient  (claude-opus-4-*, claude-sonnet-4-*, claude-haiku-4-*, …)

Quick start::

    from conduit_sdk.providers.anthropic import AnthropicLLMClient
    from conduit_sdk.core.config import ClientConfig, CostConfig
    from conduit_sdk.models.common import Message
    from conduit_sdk.models.requests import LLMRequest

    client = AnthropicLLMClient(ClientConfig(
        model="claude-sonnet-4-6",
        api_key="sk-ant-...",         # or set ANTHROPIC_API_KEY env var
        cost=CostConfig(
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
        ),
    ))
    response = await client.generate(LLMRequest(messages=[
        Message.system("You are a concise assistant."),
        Message.user("What is the speed of light?"),
    ]))
    print(response.content)
"""

from conduit_sdk.providers.anthropic.llm import AnthropicLLMClient

__all__ = ["AnthropicLLMClient"]
