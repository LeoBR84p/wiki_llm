"""Tests for LLM backend clients — OpenAI, Ollama, OpenRouter, Bedrock."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm.base import LLMResponse
from src.models.config import LLMConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_cfg(backend: str = "openai", model: str = "gpt-4o") -> LLMConfig:
    return LLMConfig(backend=backend, model_id=model)


def _fake_openai_response(text: str = "hello", tokens_in: int = 10, tokens_out: int = 5):
    usage = SimpleNamespace(
        prompt_tokens=tokens_in,
        completion_tokens=tokens_out,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIClient:
    def test_init_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from src.llm.openai import OpenAIClient
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIClient(_llm_cfg("openai"))

    def test_init_succeeds_with_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("src.llm.openai.AsyncOpenAI"), patch("src.llm.openai.instructor"):
            from importlib import reload
            import src.llm.openai as mod
            reload(mod)
            client = mod.OpenAIClient(_llm_cfg())
            assert client._cfg.model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_call_returns_llm_response(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake_resp = _fake_openai_response("Hello world")

        with patch("src.llm.openai.AsyncOpenAI") as MockAsync, \
             patch("src.llm.openai.instructor"):
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)
            MockAsync.return_value = mock_client

            from importlib import reload
            import src.llm.openai as mod
            reload(mod)
            client = mod.OpenAIClient(_llm_cfg())
            client._client = mock_client

            result = await client.call("system", "user")
            assert isinstance(result, LLMResponse)
            assert result.text == "Hello world"
            assert result.tokens_in == 10
            assert result.tokens_out == 5


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class TestOllamaClient:
    def test_init_uses_env_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434/v1")
        with patch("src.llm.ollama.AsyncOpenAI") as MockAsync, \
             patch("src.llm.ollama.instructor"):
            MockAsync.return_value = MagicMock()
            from src.llm.ollama import OllamaClient
            OllamaClient(_llm_cfg("ollama", "llama3"))
            assert MockAsync.called
            kwargs = MockAsync.call_args.kwargs
            assert kwargs["base_url"] == "http://custom:11434/v1"

    def test_init_default_base_url(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        with patch("src.llm.ollama.AsyncOpenAI") as MockAsync, \
             patch("src.llm.ollama.instructor"):
            MockAsync.return_value = MagicMock()
            from src.llm.ollama import OllamaClient
            OllamaClient(_llm_cfg("ollama", "llama3"))
            assert MockAsync.called
            kwargs = MockAsync.call_args.kwargs
            assert "11434" in kwargs["base_url"]

    @pytest.mark.asyncio
    async def test_call_returns_llm_response(self):
        fake_resp = _fake_openai_response("Ollama response")
        # ollama doesn't have cached_tokens in usage
        usage = SimpleNamespace(prompt_tokens=8, completion_tokens=4)
        fake_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ollama response"))],
            usage=usage,
        )

        with patch("src.llm.ollama.AsyncOpenAI") as MockAsync, \
             patch("src.llm.ollama.instructor"):
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)
            MockAsync.return_value = mock_client

            from importlib import reload
            import src.llm.ollama as mod
            reload(mod)
            client = mod.OllamaClient(_llm_cfg("ollama", "llama3"))
            client._client = mock_client

            result = await client.call("system", "user")
            assert isinstance(result, LLMResponse)
            assert result.text == "Ollama response"


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


class TestOpenRouterClient:
    def test_init_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_APIKEY", raising=False)
        from src.llm.openrouter import OpenRouterClient
        with pytest.raises(RuntimeError, match="OPENROUTER_APIKEY"):
            OpenRouterClient(_llm_cfg("openrouter"))

    def test_init_succeeds_with_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_APIKEY", "or-test-key")
        with patch("src.llm.openrouter.AsyncOpenAI"), \
             patch("src.llm.openrouter.instructor"):
            from importlib import reload
            import src.llm.openrouter as mod
            reload(mod)
            client = mod.OpenRouterClient(_llm_cfg("openrouter", "mistral-7b"))
            assert client._cfg.model_id == "mistral-7b"

    @pytest.mark.asyncio
    async def test_call_returns_llm_response(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_APIKEY", "or-test-key")
        usage = SimpleNamespace(
            prompt_tokens=6, completion_tokens=3, prompt_tokens_details=None
        )
        fake_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OR response"))],
            usage=usage,
        )

        with patch("src.llm.openrouter.AsyncOpenAI") as MockAsync, \
             patch("src.llm.openrouter.instructor"):
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)
            MockAsync.return_value = mock_client

            from importlib import reload
            import src.llm.openrouter as mod
            reload(mod)
            client = mod.OpenRouterClient(_llm_cfg("openrouter", "m7b"))
            client._client = mock_client

            result = await client.call("system", "user")
            assert isinstance(result, LLMResponse)
            assert result.text == "OR response"


# ---------------------------------------------------------------------------
# Bedrock — pure functions
# ---------------------------------------------------------------------------


class TestBedrockHelpers:
    def test_extract_text_normal(self):
        from src.llm.bedrock import _extract_text
        resp = {
            "output": {
                "message": {
                    "content": [{"text": "Hello"}, {"text": " world"}]
                }
            }
        }
        assert _extract_text(resp) == "Hello world"

    def test_extract_text_empty(self):
        from src.llm.bedrock import _extract_text
        assert _extract_text({}) == ""

    def test_extract_text_no_text_blocks(self):
        from src.llm.bedrock import _extract_text
        resp = {"output": {"message": {"content": [{"type": "image"}]}}}
        assert _extract_text(resp) == ""

    def test_extract_usage_normal(self):
        from src.llm.bedrock import _extract_usage
        resp = {"usage": {"inputTokens": 15, "outputTokens": 7}}
        tin, tout = _extract_usage(resp)
        assert tin == 15
        assert tout == 7

    def test_extract_usage_missing(self):
        from src.llm.bedrock import _extract_usage
        tin, tout = _extract_usage({})
        assert tin is None
        assert tout is None

    def test_sync_call_success(self):
        from src.llm.bedrock import _sync_call
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Bedrock reply"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = _sync_call(mock_client, "model-id", 0.2, 1024, "sys", "user")
        assert isinstance(result, LLMResponse)
        assert result.text == "Bedrock reply"
        assert result.tokens_in == 10
        assert result.tokens_out == 5
        assert result.attempts == 1

    def test_sync_call_retries_and_raises(self):
        from src.llm.bedrock import _sync_call
        mock_client = MagicMock()
        mock_client.converse.side_effect = ConnectionError("timeout")
        with pytest.raises(RuntimeError, match="Bedrock failed"):
            _sync_call(mock_client, "m", 0.0, 100, "s", "u", max_attempts=2)
        assert mock_client.converse.call_count == 2


class TestBedrockClient:
    @pytest.mark.asyncio
    async def test_call_delegates_to_thread(self, monkeypatch):
        with patch("src.llm.bedrock.boto3") as mock_boto, \
             patch("src.llm.bedrock.Config"):
            mock_boto.client.return_value = MagicMock()

            from importlib import reload
            import src.llm.bedrock as mod
            reload(mod)

            client = mod.BedrockClient(_llm_cfg("bedrock", "anthropic.claude-3"))
            fake_resp = LLMResponse(
                text="AWS response", tokens_in=10, tokens_out=5,
                cached_tokens=None, model_id="anthropic.claude-3", attempts=1,
            )
            with patch("src.llm.bedrock._sync_call", return_value=fake_resp):
                result = await client.call("sys", "user")
            assert result.text == "AWS response"
