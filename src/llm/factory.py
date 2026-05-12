"""Factory that instantiates the correct LLM backend client from configuration.

Keeps backend-specific imports lazy so that projects which use only one backend
(e.g. only OpenRouter) do not need the others installed.
"""

from __future__ import annotations

from ..models.config import LLMConfig
from .base import BaseLLMClient


def create_client(config: LLMConfig) -> BaseLLMClient:
    """Instantiate the LLM backend client specified in the configuration.

    Uses a structural match on ``config.backend`` to import and return the
    appropriate client class.  Each import is deferred to this function so
    that optional backend dependencies (boto3, openai SDK, etc.) are only
    imported when actually needed.

    Args:
        config: An LLMConfig instance with at least ``backend`` and ``model_id``
            fields populated.  Additional fields (api_key, region, etc.) are
            read by the individual client constructors.

    Returns:
        A BaseLLMClient instance ready to accept ``call`` and ``call_structured``
        invocations.

    Raises:
        ValueError: If ``config.backend`` is not one of the supported values.
    """
    match config.backend:
        case "bedrock":
            from .bedrock import BedrockClient  # noqa: PLC0415
            return BedrockClient(config)
        case "openrouter":
            from .openrouter import OpenRouterClient  # noqa: PLC0415
            return OpenRouterClient(config)
        case "openai":
            from .openai import OpenAIClient  # noqa: PLC0415
            return OpenAIClient(config)
        case "ollama":
            from .ollama import OllamaClient  # noqa: PLC0415
            return OllamaClient(config)
        case _:
            raise ValueError(f"Backend desconhecido: {config.backend}")
