"""
MyModelClient — async HTTP adapter for the MyModel API.

This module is the only place that knows how to speak to your API.
Everything else (retry, rate-limiting, cost tracking, the fluent builder)
is inherited from conduit-sdk.

Adapt the two sections marked  ←── ADAPT THIS  to match your actual
API request/response shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from conduit_sdk.clients import LLMClient
from conduit_sdk.core.config import ClientConfig, CostConfig
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason, LLMResponse
from mymodel_client.config import MyModelConfig


class MyModelClient(LLMClient):
    """
    Async client for the MyModel API.

    Usage::

        client = MyModelClient(MyModelConfig(
            base_url="https://api.mycompany.com",
            api_key="sk-...",
        ))

        # fluent builder
        response = await client.generate(
            MyModelClient.Request.Builder()
            .system("Be concise.")
            .user("Explain fine-tuning.")
            .max_tokens(150)
            .build()
        )
        print(response.content)
        print(f"tokens used: {response.usage.total_tokens}")

        # or direct construction
        from conduit_sdk.models.requests import LLMRequest
        from conduit_sdk.models.common import Message
        response = await client.generate(
            LLMRequest(messages=[Message.user("Hello")])
        )
    """

    # Expose LLMRequest as MyModelClient.Request so callers
    # don't need to import conduit_sdk directly.
    Request = LLMRequest

    def __init__(self, my_config: MyModelConfig) -> None:
        # Translate MyModelConfig → conduit-sdk ClientConfig so the
        # middleware stack (retry, cost tracker, logger) is configured.
        sdk_config = ClientConfig(
            provider="mymodel",
            model=my_config.model,
            api_key=my_config.api_key,
            max_retries=my_config.max_retries,
            cost=CostConfig(
                input_cost_per_1k_tokens=my_config.input_cost_per_1k_tokens,
                output_cost_per_1k_tokens=my_config.output_cost_per_1k_tokens,
            ),
        )
        super().__init__(sdk_config)
        self._my_config = my_config

        # One shared httpx client for the lifetime of this object.
        # This keeps the TCP connection pool alive across requests.
        self._http = httpx.AsyncClient(
            base_url=my_config.base_url,
            headers={
                "Authorization": f"Bearer {my_config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=my_config.timeout,
        )

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Async context manager support  (optional but handy in scripts)
    #
    #   async with MyModelClient(cfg) as client:
    #       response = await client.generate(...)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MyModelClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # ←── ADAPT THIS: build the request payload your API expects
    # ------------------------------------------------------------------

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """
        Translate a conduit-sdk LLMRequest into the JSON body your API expects.

        Default shape matches a standard OpenAI-compatible API.
        Change the keys/structure to match your own API contract.
        """
        payload: dict[str, Any] = {
            "model": request.model or self._my_config.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        # pass any provider-specific extras through as-is
        payload.update(request.extra)
        return payload

    # ------------------------------------------------------------------
    # ←── ADAPT THIS: parse the response your API returns
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """
        Translate your API's JSON response into a conduit-sdk LLMResponse.

        Default parsing matches a standard OpenAI-compatible response shape.
        Adjust field paths to match your own API contract.
        """
        # --- content ---
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason") or "stop"

        # --- usage ---
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=content),
            finish_reason=FinishReason(finish_reason),
            usage=usage,
            model=data.get("model", self._my_config.model),
            provider="mymodel",
            raw_response=data,
        )

    # ------------------------------------------------------------------
    # conduit-sdk abstract method implementations
    # ------------------------------------------------------------------

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        payload = self._build_payload(request)
        response = await self._http.post("/v1/chat/completions", json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"MyModel API error {response.status_code}: {response.text}")

        return self._parse_response(response.json())

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Yield text tokens from a streaming response.

        Expects your API to stream newline-delimited JSON events in the shape:
            data: {"choices": [{"delta": {"content": "token"}}]}
            data: [DONE]

        Adjust the parsing if your API uses a different streaming format.
        """
        payload = {**self._build_payload(request), "stream": True}

        async with self._http.stream("POST", "/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"MyModel API stream error {resp.status_code}: {body.decode()}")

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                import json  # noqa: PLC0415

                data = json.loads(chunk)
                delta = data["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
