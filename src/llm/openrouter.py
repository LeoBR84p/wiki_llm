"""Async OpenRouter client for the wiki-llm pipeline.

Uses the OpenAI-compatible SDK pointing at the OpenRouter base URL.
Reads OPENROUTER_APIKEY and optionally OPENROUTER_BASE_URL from the environment.
"""

from __future__ import annotations

import os
from typing import Any

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..models.config import LLMConfig
from .base import LLMResponse


class OpenRouterClient:
    """LLM client backed by the OpenRouter API (OpenAI-compatible).

    Attributes:
        _cfg: LLMConfig with model_id, temperature, and max_tokens.
        _client: AsyncOpenAI instance pointed at the OpenRouter base URL.
        _instructor: Instructor-patched client (JSON mode) for structured extraction.
    """
    def __init__(self, config: LLMConfig) -> None:
        """Initialize the OpenRouter client.

        Args:
            config: LLMConfig specifying model, temperature, and max_tokens.

        Raises:
            RuntimeError: If OPENROUTER_APIKEY is not set in the environment.
        """
        self._cfg = config
        api_key = os.environ.get("OPENROUTER_APIKEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_APIKEY ausente no .env.")
        base_url = os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._instructor = instructor.from_openai(self._client, mode=instructor.Mode.JSON)

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
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else None,
            tokens_out=usage.completion_tokens if usage else None,
            cached_tokens=getattr(usage, "prompt_tokens_details", None)
            and getattr(usage.prompt_tokens_details, "cached_tokens", None),
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
