# mymodel-client

Async Python client library for the MyModel API.

Built on [conduit-sdk](https://github.com/erenat77/llm-conduit) — gives every caller retry, rate-limiting, cost tracking, and a fluent request builder with no extra setup.

---

## Installation

```bash
pip install mymodel-client
```

**Requirements:** Python 3.11+

---

## Quick start

```python
import asyncio
from mymodel_client import MyModelClient, MyModelConfig

client = MyModelClient(MyModelConfig(
    base_url="https://api.mycompany.com",
    api_key="sk-...",
))

async def main():
    response = await client.generate(
        MyModelClient.Request.Builder()
        .system("You are a concise assistant.")
        .user("What is RLHF?")
        .max_tokens(150)
        .temperature(0.3)
        .build()
    )
    print(response.content)
    print(f"tokens used: {response.usage.total_tokens}")
    await client.close()

asyncio.run(main())
```

---

## Configuration

All options are passed via `MyModelConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | required | Root URL of your model API |
| `api_key` | `str` | required | Bearer token for authentication |
| `model` | `str` | `"mymodel-v1"` | Default model name |
| `timeout` | `float` | `30.0` | HTTP timeout in seconds |
| `max_retries` | `int` | `3` | Retries on transient failures |
| `input_cost_per_1k_tokens` | `float` | `0.0` | Used by cost tracker |
| `output_cost_per_1k_tokens` | `float` | `0.0` | Used by cost tracker |

---

## Usage

### Single-turn completion

```python
response = await client.generate(
    MyModelClient.Request.Builder()
    .user("Summarise this document in 3 bullet points.")
    .max_tokens(200)
    .build()
)
print(response.content)
```

### Multi-turn conversation

```python
from conduit_sdk.models.common import Message

history = [
    Message.system("You are a helpful assistant."),
    Message.user("What is fine-tuning?"),
    Message.assistant("Fine-tuning adapts a pretrained model to a specific task..."),
    Message.user("How does it differ from RLHF?"),
]

response = await client.generate(
    MyModelClient.Request.Builder()
    .messages(*history)          # pass a list of Message objects
    .max_tokens(300)
    .build()
)
```

### Streaming

```python
request = (
    MyModelClient.Request.Builder()
    .user("Explain transformers step by step.")
    .max_tokens(400)
    .stream(True)
    .build()
)

async for chunk in client.stream(request):
    print(chunk, end="", flush=True)
```

### Cost tracking

```python
client = MyModelClient(MyModelConfig(
    base_url="https://api.mycompany.com",
    api_key="sk-...",
    input_cost_per_1k_tokens=0.001,
    output_cost_per_1k_tokens=0.002,
))

response = await client.generate(...)
if response.cost:
    print(f"cost: ${response.cost.total_cost:.6f}")
```

### Async context manager

```python
async with MyModelClient(MyModelConfig(...)) as client:
    response = await client.generate(...)
# connection pool closed automatically
```

---

## Using inside FastAPI

Create the client once at startup via `lifespan` and inject it into routes with `Depends`. Never create a new client per request — the shared httpx connection pool is what makes this efficient.

```python
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from mymodel_client import MyModelClient, MyModelConfig

_client: MyModelClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = MyModelClient(MyModelConfig(
        base_url="https://api.mycompany.com",
        api_key="sk-...",
    ))
    yield
    await _client.close()

app = FastAPI(lifespan=lifespan)

def get_client() -> MyModelClient:
    return _client

@app.post("/summarise")
async def summarise(body: dict, client: MyModelClient = Depends(get_client)):
    response = await client.generate(
        MyModelClient.Request.Builder()
        .system("Summarise in 2 sentences.")
        .user(body["text"])
        .max_tokens(120)
        .build()
    )
    return {"summary": response.content}
```

---

## Running the tests

```bash
pip install "mymodel-client[dev]"
pytest tests/
```

Tests use [respx](https://lundberg.github.io/respx/) to mock HTTP calls — no real API key or network access required.

---

## Adapting to your API

The only file you need to modify is `mymodel_client/client.py`. Two methods control the entire HTTP contract:

**`_build_payload(request)`** — translates a conduit-sdk `LLMRequest` into the JSON body your API expects.

**`_parse_response(data)`** — translates your API's JSON response back into an `LLMResponse`.

Everything else (retry logic, rate limiting, cost tracking, the fluent builder) is inherited from conduit-sdk and requires no changes.

---

## License

MIT
