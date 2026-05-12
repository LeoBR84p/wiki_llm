"""Async AWS Bedrock Converse client for the wiki-llm pipeline.

Boto3 is synchronous, so all calls are wrapped in asyncio.to_thread.
Reads AWS credentials and region from environment variables.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import boto3
from botocore.config import Config
from pydantic import BaseModel

from ..models.config import LLMConfig
from .base import LLMResponse


def _make_boto3_client(region: str) -> Any:
    """Create and configure a boto3 bedrock-runtime client.

    Optionally sets AWS_BEARER_TOKEN_BEDROCK from the WIKI_BEDROCK_LOGINKEY
    environment variable for SSO-style authentication.  Uses adaptive retries
    with a 90 s read timeout.

    Args:
        region: AWS region name (e.g. ``"sa-east-1"``).

    Returns:
        A boto3 client for the ``bedrock-runtime`` service.
    """
    loginkey = os.environ.get("AWS_LOGINKEY", "").strip()
    if loginkey:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = loginkey
    cfg = Config(
        connect_timeout=10,
        read_timeout=90,
        retries={"max_attempts": 1, "mode": "adaptive"},
    )
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def _extract_text(resp: dict[str, Any]) -> str:
    """Extract the plain text content from a Bedrock Converse response.

    Args:
        resp: The raw dict response from boto3 client.converse().

    Returns:
        Concatenated text from all text content blocks, or empty string.
    """
    output = resp.get("output", {})
    message = output.get("message", {}) if isinstance(output, dict) else {}
    content = message.get("content", []) if isinstance(message, dict) else []
    return "".join(b["text"] for b in content if isinstance(b, dict) and "text" in b)


def _extract_usage(resp: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract input and output token counts from a Bedrock Converse response.

    Args:
        resp: The raw dict response from boto3 client.converse().

    Returns:
        A (tokens_in, tokens_out) tuple; either value may be None if missing.
    """
    usage = resp.get("usage", {})
    return usage.get("inputTokens"), usage.get("outputTokens")


def _sync_call(client: Any, model_id: str, temperature: float, max_tokens: int,
               system: str, user: str, max_attempts: int = 3) -> LLMResponse:
    """Synchronously call Bedrock Converse with exponential backoff retries.

    Called from a thread via asyncio.to_thread by BedrockClient.call().

    Args:
        client: A boto3 bedrock-runtime client.
        model_id: Bedrock model identifier string.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        system: System prompt text.
        user: User message text.
        max_attempts: Maximum number of retry attempts on transient failure.

    Returns:
        An LLMResponse with the generated text and token usage.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    messages = [{"role": "user", "content": [{"text": user}]}]
    system_list = [{"text": system}]
    inf_cfg = {"temperature": temperature, "maxTokens": max_tokens}
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=messages,
                system=system_list,
                inferenceConfig=inf_cfg,
            )
            tokens_in, tokens_out = _extract_usage(resp)
            return LLMResponse(
                text=_extract_text(resp),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cached_tokens=None,
                model_id=model_id,
                attempts=attempt,
                raw=resp,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Bedrock falhou após {max_attempts} tentativas") from last_exc


class BedrockClient:
    """Async LLM client backed by AWS Bedrock Converse.

    All boto3 calls run in a thread pool via asyncio.to_thread to avoid
    blocking the event loop.

    Attributes:
        _cfg: LLMConfig with model_id, temperature, and max_tokens.
        _boto: Configured boto3 bedrock-runtime client.
    """
    def __init__(self, config: LLMConfig) -> None:
        """Initialize the Bedrock client.

        Args:
            config: LLMConfig specifying model_id, temperature, and max_tokens.
        """
        self._cfg = config
        region = os.environ.get("WIKI_BEDROCK_REGION", "sa-east-1")
        self._boto = _make_boto3_client(region)

    async def call(self, system: str, user: str) -> LLMResponse:
        return await asyncio.to_thread(
            _sync_call,
            self._boto,
            self._cfg.model_id,
            self._cfg.temperature,
            self._cfg.max_tokens,
            system,
            user,
        )

    async def call_structured(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        import instructor  # noqa: PLC0415
        patched = instructor.from_boto3(self._boto)
        return await asyncio.to_thread(
            patched.converse,
            modelId=self._cfg.model_id,
            messages=[{"role": "user", "content": [{"text": user}]}],
            system=[{"text": system}],
            response_model=schema,
        )
