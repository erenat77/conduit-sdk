"""
MyModelConfig — typed configuration for the MyModel API.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MyModelConfig:
    """
    Connection settings for the MyModel API.

    Parameters
    ----------
    base_url:
        Root URL of your model API, e.g. ``"https://api.mycompany.com"``.
    api_key:
        Bearer token for authentication.
    model:
        Default model name sent to the API.  Can be overridden per-request.
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        How many times to retry on transient failures (handled by conduit-sdk
        middleware).
    input_cost_per_1k_tokens:
        Optional cost metadata — used by conduit-sdk's cost tracker.
    output_cost_per_1k_tokens:
        Optional cost metadata.
    """

    base_url: str
    api_key: str
    model: str = "mymodel-v1"
    timeout: float = 30.0
    max_retries: int = 3
    input_cost_per_1k_tokens: float = 0.0
    output_cost_per_1k_tokens: float = 0.0
