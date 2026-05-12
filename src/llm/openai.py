"""Async OpenAI client for the wiki-llm pipeline.

Wraps AsyncOpenAI for plain text calls and instructor for structured
extraction.  Reads OPENAI_API_KEY from the environment.
"""

from __future__ import annotations

import os

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..models.config import LLMConfig
from .base import LLMResponse


class OpenAIClient:
    """LLM client backed by the OpenAI Chat Completions API.

    Attributes:
        _cfg: LLMConfig with model_id, temperature, and max_tokens.
        _client: Underlying AsyncOpenAI instance.
        _instructor: Instructor-patched client for structured extraction.
    """
    def __init__(self, config: LLMConfig) -> None:
        """Initialize the OpenAI client.

        Args:
            config: LLMConfig specifying model, temperature, and max_tokens.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set in the environment.
        """
        self._cfg = config
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY ausente no .env.")
        self._client = AsyncOpenAI(api_key=api_key)
        self._instructor = instructor.from_openai(self._client)

    async def call(self, system: str, user: str) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=self._cfg.model_id,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        cached = None
        if usage and hasattr(usage, "prompt_tokens_details"):
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else None,
            tokens_out=usage.completion_tokens if usage else None,
            cached_tokens=cached,
            model_id=self._cfg.model_id,
            attempts=1,
        )

    async def call_structured(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        return await self._instructor.chat.completions.create(
            model=self._cfg.model_id,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            response_model=schema,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
