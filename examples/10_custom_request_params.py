"""
Example 10 — Custom request parameters with ProviderParams
===========================================================
Shows:
  - Defining typed provider-specific parameters via ProviderParams
  - Mixing them into any base request with multiple inheritance
  - Building an extended builder that returns the custom type
  - Wiring the extended builder onto Request.Builder so callers
    get a consistent API: MyRequest.Builder().field().build()
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from conduit_sdk.clients import LLMClient
from conduit_sdk.core.config import ClientConfig
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest, LLMRequestBuilder, ProviderParams
from conduit_sdk.models.responses import FinishReason, LLMResponse

# ---------------------------------------------------------------------------
# 1. Declare typed provider-specific parameters
# ---------------------------------------------------------------------------

class OpenAIParams(ProviderParams):
    """
    OpenAI-specific parameters not covered by the standard LLMRequest schema.

    Every field must have a default value — these params are optional
    extensions on top of the base request.
    """
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    parallel_tool_calls: bool = True
    logprobs: bool = False
    top_logprobs: int | None = None


# ---------------------------------------------------------------------------
# 2. Mix into any base request class
# ---------------------------------------------------------------------------

class OpenAILLMRequest(LLMRequest, OpenAIParams):
    """LLMRequest extended with typed OpenAI parameters."""
    pass


# ---------------------------------------------------------------------------
# 3. Extend the matching builder to return the custom type
# ---------------------------------------------------------------------------

class OpenAILLMRequestBuilder(LLMRequestBuilder):
    """
    Fluent builder that produces OpenAILLMRequest instances.

    Inherits all standard LLMRequest builder methods (.system, .user,
    .max_tokens, .temperature, etc.) and adds OpenAI-specific ones.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reasoning_effort: str = "medium"
        self._parallel_tool_calls: bool = True
        self._logprobs: bool = False
        self._top_logprobs: int | None = None

    def reasoning_effort(self, value: Literal["low", "medium", "high"]) -> OpenAILLMRequestBuilder:
        """Set the reasoning effort level for o1/o3 models."""
        self._reasoning_effort = value
        return self

    def parallel_tool_calls(self, enabled: bool) -> OpenAILLMRequestBuilder:
        """Enable or disable parallel tool call execution."""
        self._parallel_tool_calls = enabled
        return self

    def logprobs(self, enabled: bool = True) -> OpenAILLMRequestBuilder:
        """Request log-probability output alongside the completion."""
        self._logprobs = enabled
        return self

    def top_logprobs(self, n: int) -> OpenAILLMRequestBuilder:
        """Return the top-N log-probabilities for each token position."""
        self._top_logprobs = n
        return self

    def build(self) -> OpenAILLMRequest:  # type: ignore[override]
        """Produce the immutable OpenAILLMRequest."""
        base = super().build()
        return OpenAILLMRequest(
            **base.model_dump(),
            reasoning_effort=self._reasoning_effort,
            parallel_tool_calls=self._parallel_tool_calls,
            logprobs=self._logprobs,
            top_logprobs=self._top_logprobs,
        )


# ---------------------------------------------------------------------------
# 4. Wire the extended builder onto the custom request class
#    → callers use: OpenAILLMRequest.Builder().reasoning_effort("high").build()
# ---------------------------------------------------------------------------

OpenAILLMRequest.Builder = OpenAILLMRequestBuilder


# ---------------------------------------------------------------------------
# 5. Mock client that prints the custom params to show they're passed through
# ---------------------------------------------------------------------------

class MockOpenAIClient(LLMClient):

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        last = next(
            m.content for m in reversed(request.messages) if m.role == MessageRole.USER
        )
        # The provider adapter can read typed fields if the request is extended
        effort = getattr(request, "reasoning_effort", "n/a")
        logprobs = getattr(request, "logprobs", False)
        reply = f"[mock-openai] '{last}' | effort={effort} logprobs={logprobs}"
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=reply),
            finish_reason=FinishReason.STOP,
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model=self.config.model,
            provider=self.config.provider,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        response = await self._generate(request)
        for word in response.content.split():
            yield word + " "


client = MockOpenAIClient(config=ClientConfig(provider="openai", model="o3-mini"))


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("Example 10 — Custom request parameters")
    print("=" * 60)

    # ── Standard usage: base LLMRequest unchanged ────────────────────────────
    print("\n[1] Standard LLMRequest.Builder() — no custom params")
    req = (
        LLMRequest.Builder()
        .system("Be concise.")
        .user("What is attention in transformers?")
        .max_tokens(120)
        .build()
    )
    r = await client.generate(req)
    print(f"    {r.content}")

    # ── Extended usage: OpenAILLMRequest.Builder() ────────────────────────────
    print("\n[2] OpenAILLMRequest.Builder() — with typed OpenAI params")
    req2 = (
        OpenAILLMRequest.Builder()
        .system("You are a precise reasoning assistant.")
        .user("Explain RLHF step by step.")
        .max_tokens(400)
        .temperature(0.2)
        .reasoning_effort("high")
        .logprobs(True)
        .top_logprobs(5)
        .build()
    )
    assert isinstance(req2, OpenAILLMRequest)
    assert isinstance(req2, LLMRequest)          # still a base LLMRequest
    assert req2.reasoning_effort == "high"
    assert req2.logprobs is True
    r2 = await client.generate(req2)
    print(f"    {r2.content}")

    # ── Direct construction still works ─────────────────────────────────────
    print("\n[3] Direct construction — mixed params")
    req3 = OpenAILLMRequest(
        messages=[Message.user("Summarise chain-of-thought prompting.")],
        max_tokens=200,
        reasoning_effort="low",
        parallel_tool_calls=False,
    )
    r3 = await client.generate(req3)
    print(f"    {r3.content}")

    # ── model_dump() includes all fields ────────────────────────────────────
    print("\n[4] model_dump() includes base + custom fields")
    dumped = req2.model_dump()
    for key in ("messages", "max_tokens", "reasoning_effort", "logprobs", "top_logprobs"):
        print(f"    {key}: {dumped[key]!r}")

    print("\n✓ All assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
