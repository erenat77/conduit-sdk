"""
Shared value objects used across request and response models.
"""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        pass


from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single turn in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    # Raw extra fields from provider responses (tool calls, attachments, etc.)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=MessageRole.ASSISTANT, content=content)


class Usage(BaseModel):
    """Token / unit consumption reported by the provider."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Video / image / embedding specifics
    image_count: int = 0
    video_seconds: float = 0.0
    embedding_count: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            image_count=self.image_count + other.image_count,
            video_seconds=self.video_seconds + other.video_seconds,
            embedding_count=self.embedding_count + other.embedding_count,
        )


class Cost(BaseModel):
    """Estimated monetary cost for a single call (USD)."""

    model_config = ConfigDict(frozen=True)

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
            currency=self.currency,
        )
