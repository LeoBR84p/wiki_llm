"""Async Ollama client for the wiki-llm pipeline.

Uses the OpenAI-compatible API exposed by Ollama at OLLAMA_BASE_URL
(default: http://localhost:11434/v1). No API key is required.
"""

from __future__ import annotations

import os

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..models.config import LLMConfig
from .base import LLMResponse


class OllamaClient:
    """LLM client backed by a local Ollama server (OpenAI-compatible API)."""

    def __init__(self, config: LLMConfig) -> None:
        self._cfg = config
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self._client = AsyncOpenAI(api_key="ollama", base_url=base_url, timeout=120.0)
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
            cached_tokens=None,
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
