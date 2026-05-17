"""
Unit tests for MyModelClient — no real HTTP calls.

Uses respx to mock the httpx transport so tests run offline.

Install test deps:
    pip install pytest pytest-asyncio respx
"""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response
from mymodel_client import MyModelClient, MyModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_URL = "https://api.mycompany.com"


@pytest.fixture
def cfg() -> MyModelConfig:
    return MyModelConfig(
        base_url=BASE_URL,
        api_key="sk-test",
        model="mymodel-v1",
    )


CHAT_RESPONSE = {
    "id": "chatcmpl-abc123",
    "model": "mymodel-v1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "RLHF aligns models with human feedback."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
}


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_generate_returns_content(cfg: MyModelConfig) -> None:
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=Response(200, json=CHAT_RESPONSE)
    )

    async with MyModelClient(cfg) as client:
        response = await client.generate(
            MyModelClient.Request.Builder().user("What is RLHF?").max_tokens(100).build()
        )

    assert response.content == "RLHF aligns models with human feedback."
    assert response.usage.total_tokens == 21
    assert response.provider == "mymodel"
    assert response.model == "mymodel-v1"


@pytest.mark.asyncio
@respx.mock
async def test_generate_sends_correct_payload(cfg: MyModelConfig) -> None:
    route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=Response(200, json=CHAT_RESPONSE)
    )

    async with MyModelClient(cfg) as client:
        await client.generate(
            MyModelClient.Request.Builder()
            .system("Be concise.")
            .user("Explain attention.")
            .temperature(0.3)
            .max_tokens(150)
            .build()
        )

    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][0] == {"role": "system", "content": "Be concise."}
    assert sent["messages"][1] == {"role": "user", "content": "Explain attention."}
    assert sent["temperature"] == pytest.approx(0.3)
    assert sent["max_tokens"] == 150


@pytest.mark.asyncio
@respx.mock
async def test_api_error_raises(cfg: MyModelConfig) -> None:
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=Response(500, text="Internal Server Error")
    )

    async with MyModelClient(cfg) as client:
        with pytest.raises(RuntimeError, match="500"):
            await client.generate(MyModelClient.Request.Builder().user("Hello").build())


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

SSE_STREAM = "\n".join(
    [
        'data: {"choices":[{"delta":{"content":"RLHF "}}]}',
        'data: {"choices":[{"delta":{"content":"aligns "}}]}',
        'data: {"choices":[{"delta":{"content":"models."}}]}',
        "data: [DONE]",
    ]
)


@pytest.mark.asyncio
@respx.mock
async def test_stream_yields_chunks(cfg: MyModelConfig) -> None:
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=Response(200, text=SSE_STREAM, headers={"Content-Type": "text/event-stream"})
    )

    chunks: list[str] = []
    async with MyModelClient(cfg) as client:
        request = MyModelClient.Request.Builder().user("What is RLHF?").stream(True).build()
        async for chunk in client.stream(request):
            chunks.append(chunk)

    assert chunks == ["RLHF ", "aligns ", "models."]
    assert "".join(chunks) == "RLHF aligns models."
