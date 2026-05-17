"""
Example 01 — Basic LLM chat completion
=======================================
Shows:
  - Implementing a minimal LLMClient subclass (mock provider)
  - Building a multi-turn conversation (direct construction and fluent builder)
  - Async generate() and its sync wrapper
  - Accessing response fields: content, usage, finish_reason
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from conduit_sdk.clients import LLMClient
from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest, LLMRequestBuilder
from conduit_sdk.models.responses import FinishReason, LLMResponse

# ---------------------------------------------------------------------------
# 1. Implement your provider adapter
#    → Override _generate (required) and _stream (required by ABC)
# ---------------------------------------------------------------------------


class MockChatClient(LLMClient):
    """
    Toy LLM that echoes the last user message back with a prefix.
    Replace _generate with a real SDK call (openai, anthropic, etc.).
    """

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        last_user_msg = next(
            m.content for m in reversed(request.messages) if m.role == MessageRole.USER
        )
        reply = f"[mock] You said: '{last_user_msg}'. Here is a thoughtful response."

        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=reply),
            finish_reason=FinishReason.STOP,
            usage=Usage(
                prompt_tokens=len(" ".join(m.content for m in request.messages).split()),
                completion_tokens=len(reply.split()),
                total_tokens=len(" ".join(m.content for m in request.messages).split())
                + len(reply.split()),
            ),
            model=self.config.model,
            provider=self.config.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        # Required by the ABC — used in example 02
        response = await self._generate(request)
        for word in response.content.split():
            yield word + " "


# ---------------------------------------------------------------------------
# 2. Configure the client
# ---------------------------------------------------------------------------

config = ClientConfig(
    provider="mock",
    model="mock-llm-v1",
    cost=CostConfig(
        input_cost_per_1k_tokens=0.005,
        output_cost_per_1k_tokens=0.015,
    ),
)

client = MockChatClient(config=config)


# ---------------------------------------------------------------------------
# 3. Run a multi-turn conversation
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 55)
    print("Example 01 — Basic LLM Chat")
    print("=" * 55)

    # ── Option A: classic keyword-argument construction ──────────────────────
    messages = [
        Message.system("You are a concise, helpful assistant."),
        Message.user("What is the capital of France?"),
    ]
    request = LLMRequest(messages=messages, max_tokens=200, temperature=0.7)
    response = await client.generate(request)

    print(f"\nAssistant: {response.content}")
    print(f"\nFinish reason : {response.finish_reason}")
    print(f"Prompt tokens : {response.usage.prompt_tokens}")
    print(f"Output tokens : {response.usage.completion_tokens}")
    print(f"Total tokens  : {response.usage.total_tokens}")

    if response.cost:
        print(f"Estimated cost: ${response.cost.total_cost:.6f} USD")

    # ── Option B: fluent builder (LLMRequest.Builder()) ────────────────────────
    print("\n--- Fluent builder ---")
    builder_request = (
        LLMRequest.Builder()
        .system("You are a concise, helpful assistant.")
        .user("Name three programming languages invented before 1980.")
        .max_tokens(100)
        .temperature(0.3)
        .build()
    )
    builder_response = await client.generate(builder_request)
    print(f"Builder response: {builder_response.content}")

    # ── Multi-turn continuation ──────────────────────────────────────────────
    print("\n--- Multi-turn follow-up ---")
    messages.append(Message.assistant(response.content))
    messages.append(Message.user("And what is the population of that city?"))

    follow_up = await client.generate(LLMRequest(messages=messages))
    print(f"Follow-up: {follow_up.content}")


if __name__ == "__main__":
    asyncio.run(main())

    # Synchronous wrapper — must be called OUTSIDE an event loop
    print("\n--- Sync call (top-level, no event loop) ---")
    sync_request = LLMRequest.Builder().user("Tell me a fun fact.").max_tokens(80).build()
    sync_response = client.generate_sync(sync_request)
    print(f"Sync response: {sync_response.content}")
