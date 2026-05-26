"""
GeminiLLMClient — adapter for Google Gemini Chat API.

Uses the ``google-genai`` SDK (the new unified Google AI Python SDK).

Supported models: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, …

Install: pip install llm-conduit[gemini]
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from conduit_sdk.clients.llm import LLMClient
from conduit_sdk.core.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from conduit_sdk.models.common import Message, MessageRole, Usage
from conduit_sdk.models.requests import LLMRequest
from conduit_sdk.models.responses import FinishReason, LLMResponse

if TYPE_CHECKING:
    pass


def _require_genai():
    try:
        import google.genai as genai  # noqa: PLC0415

        return genai
    except ImportError:
        raise ImportError(
            "Gemini provider requires the google-genai package. "
            "Install it with: pip install llm-conduit[gemini]"
        ) from None


def _wrap_genai_error(exc: Exception, provider: str = "gemini") -> ProviderError:
    try:
        from google.api_core import exceptions as gapi_exc  # noqa: PLC0415

        if isinstance(exc, gapi_exc.Unauthenticated):
            return AuthenticationError(str(exc), provider=provider)
        if isinstance(exc, gapi_exc.ResourceExhausted):
            return RateLimitError(str(exc), provider=provider)
        if isinstance(exc, gapi_exc.DeadlineExceeded):
            return TimeoutError(str(exc), provider=provider)
        if isinstance(exc, gapi_exc.GoogleAPICallError):
            return ProviderError(str(exc), provider=provider)
    except ImportError:
        pass
    return ProviderError(str(exc), provider=provider)


def _map_finish_reason(raw: str | None) -> FinishReason:
    mapping = {
        "STOP": FinishReason.STOP,
        "MAX_TOKENS": FinishReason.LENGTH,
        "SAFETY": FinishReason.CONTENT_FILTER,
        "RECITATION": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(raw or "", FinishReason.UNKNOWN)


def _to_genai_contents(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Split a conduit message list into a Gemini system prompt + contents list.

    Gemini roles: 'user' | 'model' (not 'assistant').
    System messages are extracted and returned separately.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for m in messages:
        if m.role == MessageRole.SYSTEM:
            system_parts.append(m.content)
        else:
            gemini_role = "model" if m.role == MessageRole.ASSISTANT else "user"
            contents.append({"role": gemini_role, "parts": [{"text": m.content}]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


class GeminiLLMClient(LLMClient):
    """
    LLM client adapter for Google Gemini API.

    Uses the ``google-genai`` unified SDK (``google.genai``).

    Supported models:
      - ``gemini-2.0-flash``           — fast, multimodal
      - ``gemini-1.5-pro``             — best quality, 2M context
      - ``gemini-1.5-flash``           — balanced speed/quality
      - ``gemini-1.5-flash-8b``        — lightweight

    The API key is resolved in this order:
      1. ``config.api_key``
      2. ``GOOGLE_API_KEY`` environment variable

    Example::

        from conduit_sdk.providers.gemini import GeminiLLMClient
        from conduit_sdk.core.config import ClientConfig

        client = GeminiLLMClient(ClientConfig(
            provider="gemini",
            model="gemini-2.0-flash",
            api_key="AIza...",   # or set GOOGLE_API_KEY env var
        ))
        response = await client.generate(
            LLMRequest.Builder()
            .system("Be concise.")
            .user("Explain attention mechanisms.")
            .max_tokens(300)
            .build()
        )
        print(response.content)
    """

    def _api_key(self) -> str:
        return self.config.api_key or os.environ.get("GOOGLE_API_KEY", "")

    def _sdk_client(self):
        genai = _require_genai()
        return genai.Client(api_key=self._api_key())

    def _build_config(self, request: LLMRequest, system_instruction: str | None) -> dict[str, Any]:
        """Build the generation config dict (passed as-is; SDK accepts plain dicts)."""
        config_kwargs: dict[str, Any] = {}
        if request.max_tokens is not None:
            config_kwargs["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            config_kwargs["top_p"] = request.top_p
        if request.stop:
            config_kwargs["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        config_kwargs.update(request.extra)
        return config_kwargs or None  # type: ignore[return-value]

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        client = self._sdk_client()
        model = request.model or self.config.model
        system_instruction, contents = _to_genai_contents(request.messages)
        gen_config = self._build_config(request, system_instruction)

        try:
            raw = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            raise _wrap_genai_error(exc) from exc

        content_text = raw.text or ""
        finish_reason_raw = None
        if raw.candidates:
            finish_reason_raw = str(raw.candidates[0].finish_reason).split(".")[-1]

        usage = Usage(
            prompt_tokens=raw.usage_metadata.prompt_token_count if raw.usage_metadata else 0,
            completion_tokens=(
                raw.usage_metadata.candidates_token_count if raw.usage_metadata else 0
            ),
            total_tokens=raw.usage_metadata.total_token_count if raw.usage_metadata else 0,
        )

        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content=content_text),
            finish_reason=_map_finish_reason(finish_reason_raw),
            usage=usage,
            model=model,
            provider="gemini",
            raw_response=raw,
        )

    async def _stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._sdk_client()
        model = request.model or self.config.model
        system_instruction, contents = _to_genai_contents(request.messages)
        gen_config = self._build_config(request, system_instruction)

        try:
            async for chunk in await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=gen_config,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            raise _wrap_genai_error(exc) from exc
