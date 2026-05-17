"""Unit tests for src/llm/base.py."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.llm.base import BaseLLMClient, LLMResponse


class TestLLMResponse:
    def test_creation_with_all_fields(self):
        resp = LLMResponse(
            text="Hello world",
            tokens_in=100,
            tokens_out=50,
            cached_tokens=10,
            model_id="gpt-4o",
            attempts=1,
            raw={"choices": [{"message": {"content": "Hello world"}}]},
        )
        assert resp.text == "Hello world"
        assert resp.tokens_in == 100
        assert resp.tokens_out == 50
        assert resp.cached_tokens == 10
        assert resp.model_id == "gpt-4o"
        assert resp.attempts == 1
        assert "choices" in resp.raw

    def test_creation_with_none_token_fields(self):
        resp = LLMResponse(
            text="Response",
            tokens_in=None,
            tokens_out=None,
            cached_tokens=None,
            model_id="ollama/llama3",
            attempts=2,
        )
        assert resp.tokens_in is None
        assert resp.tokens_out is None
        assert resp.cached_tokens is None
        assert resp.attempts == 2

    def test_raw_defaults_to_empty_dict(self):
        resp = LLMResponse(
            text="x",
            tokens_in=0,
            tokens_out=0,
            cached_tokens=0,
            model_id="m",
            attempts=1,
        )
        assert resp.raw == {}

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(LLMResponse)


class TestBaseLLMClientProtocol:
    def test_protocol_is_runtime_checkable(self):
        # The protocol is decorated with @runtime_checkable
        assert hasattr(BaseLLMClient, "__protocol_attrs__") or isinstance(BaseLLMClient, type)

    def test_class_with_both_methods_satisfies_protocol(self):
        """A class with call and call_structured satisfies BaseLLMClient."""

        class MockClient:
            async def call(self, system: str, user: str) -> LLMResponse:
                return LLMResponse(
                    text="ok", tokens_in=1, tokens_out=1,
                    cached_tokens=0, model_id="mock", attempts=1,
                )

            async def call_structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
                return schema()

        client = MockClient()
        assert isinstance(client, BaseLLMClient)

    def test_class_missing_call_structured_fails_protocol(self):
        """A class without call_structured does not satisfy BaseLLMClient."""

        class IncompleteClient:
            async def call(self, system: str, user: str) -> LLMResponse:
                ...

        client = IncompleteClient()
        assert not isinstance(client, BaseLLMClient)

    def test_class_missing_call_fails_protocol(self):
        """A class without call does not satisfy BaseLLMClient."""

        class NoCallClient:
            async def call_structured(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
                ...

        client = NoCallClient()
        assert not isinstance(client, BaseLLMClient)
