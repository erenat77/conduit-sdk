"""
How a backend service uses mymodel-client
==========================================

Install:
    pip install mymodel-client

Then drop this pattern into any async backend service.
"""

from __future__ import annotations

import asyncio
import os

from mymodel_client import MyModelClient, MyModelConfig

# ---------------------------------------------------------------------------
# 1. Configure once (e.g. at module level or via dependency injection)
# ---------------------------------------------------------------------------

client = MyModelClient(
    MyModelConfig(
        base_url=os.environ["MYMODEL_BASE_URL"],  # e.g. https://api.mycompany.com
        api_key=os.environ["MYMODEL_API_KEY"],
        model="mymodel-v1",
        input_cost_per_1k_tokens=0.001,
        output_cost_per_1k_tokens=0.002,
    )
)


# ---------------------------------------------------------------------------
# 2. Call it — same interface regardless of what's behind the API
# ---------------------------------------------------------------------------


async def summarise(text: str) -> str:
    response = await client.generate(
        MyModelClient.Request.Builder()
        .system("You are a concise summariser. Reply in 2 sentences.")
        .user(text)
        .max_tokens(120)
        .temperature(0.3)
        .build()
    )
    return response.content


async def stream_reply(question: str) -> None:
    request = MyModelClient.Request.Builder().user(question).max_tokens(200).stream(True).build()
    async for chunk in client.stream(request):
        print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3. Use inside a FastAPI route — nothing changes about the route itself
# ---------------------------------------------------------------------------

# from fastapi import FastAPI
# app = FastAPI()
#
# @app.post("/summarise")
# async def summarise_route(body: dict) -> dict:
#     summary = await summarise(body["text"])
#     return {"summary": summary}


async def main() -> None:
    summary = await summarise(
        "Reinforcement learning from human feedback (RLHF) is a technique "
        "for training language models to follow instructions by learning from "
        "human preference data rather than purely from next-token prediction."
    )
    print("Summary:", summary)

    print("\nStreaming:")
    await stream_reply("What is a transformer in 3 sentences?")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
