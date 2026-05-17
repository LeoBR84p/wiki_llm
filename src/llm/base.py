"""Common protocol and dataclasses shared by all LLM backend clients.

Defines the BaseLLMClient Protocol so that the pipeline is backend-agnostic:
any object that implements ``call`` and ``call_structured`` can be injected
without changing pipeline code.  LLMResponse carries the raw fields returned
by the backend so that the LLMLogger can record them consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass(slots=True)
class LLMResponse:
    """Normalized response object returned by every LLM backend.

    Captures the generated text alongside token usage metrics so that the
    LLMLogger can write a consistent summary line regardless of which backend
    produced the response.

    Attributes:
        text: The model's decoded output text.
        tokens_in: Number of input (prompt) tokens consumed, or None if unknown.
        tokens_out: Number of output (completion) tokens generated, or None if unknown.
        cached_tokens: Tokens served from a prompt cache, or None if unsupported.
        model_id: Identifier of the model that produced the response.
        attempts: How many attempts were made (> 1 when instructor retried).
        raw: The unmodified API response dict for debugging.
    """

    text: str
    tokens_in: int | None
    tokens_out: int | None
    cached_tokens: int | None
    model_id: str
    attempts: int
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseLLMClient(Protocol):
    """Structural protocol that all LLM backend clients must satisfy.

    Using a Protocol (rather than an ABC) means backend classes do not need to
    import or subclass anything from this module, keeping each backend fully
    self-contained.  The pipeline receives a BaseLLMClient at runtime and
    calls these two methods without caring which backend is underneath.
    """

    async def call(self, system: str, user: str) -> LLMResponse:
        """Send a two-turn (system + user) prompt and return the raw response.

        Args:
            system: The system-role message (instructions / persona).
            user: The user-role message (the actual task or question).

        Returns:
            LLMResponse with the generated text and token metrics.
        """
        ...

    async def call_structured(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        """Send a prompt and parse the response into a Pydantic model via instructor.

        Uses the instructor library to enforce JSON output that validates against
        ``schema``.  Retries automatically on malformed responses (up to the
        backend's configured max_retries).

        Args:
            system: The system-role message.
            user: The user-role message.
            schema: A Pydantic BaseModel subclass that describes the expected output.

        Returns:
            A validated instance of ``schema``.
        """
        ...
