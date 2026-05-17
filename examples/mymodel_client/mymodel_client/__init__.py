"""
mymodel-client
==============
Thin async client library for the MyModel API.

Built on conduit-sdk — gives callers retry, rate-limiting, cost tracking,
and a fluent builder pattern out of the box.

Quick start::

    from mymodel_client import MyModelClient, MyModelConfig

    client = MyModelClient(MyModelConfig(
        base_url="https://api.mycompany.com",
        api_key="sk-...",
    ))

    response = await client.generate(
        MyModelClient.Request.Builder()
        .system("You are a concise assistant.")
        .user("What is RLHF?")
        .max_tokens(200)
        .build()
    )
    print(response.content)
"""

from mymodel_client.client import MyModelClient
from mymodel_client.config import MyModelConfig

__all__ = ["MyModelClient", "MyModelConfig"]
__version__ = "0.1.0"
