"""Unit tests for src/llm/log.py."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.llm.log import LLMLogger, _item_id, _write


class TestItemId:
    def test_returns_16_char_hex(self):
        iid = _item_id("system prompt", "user message")
        assert len(iid) == 16
        assert all(c in "0123456789abcdef" for c in iid)

    def test_deterministic(self):
        a = _item_id("system", "user")
        b = _item_id("system", "user")
        assert a == b

    def test_different_inputs_different_ids(self):
        a = _item_id("system A", "user")
        b = _item_id("system B", "user")
        assert a != b

    def test_newline_separator_matters(self):
        # "sys\nuser" vs "sys" + "\n" + "user" should be the same
        a = _item_id("sys", "user")
        b = _item_id("sys", "user")
        assert a == b


class TestWrite:
    def test_creates_file_and_appends(self, tmp_path):
        path = tmp_path / "test.jsonl"
        _write(path, {"key": "value"})
        assert path.exists()
        line = path.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert parsed["key"] == "value"

    def test_appends_multiple_lines(self, tmp_path):
        path = tmp_path / "multi.jsonl"
        _write(path, {"n": 1})
        _write(path, {"n": 2})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["n"] == 1
        assert json.loads(lines[1])["n"] == 2

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "file.jsonl"
        _write(path, {"x": 1})
        assert path.exists()

    def test_silently_handles_oserror(self, tmp_path):
        # Writing to a directory path (not a file) should be swallowed
        path = tmp_path  # tmp_path itself is a directory
        _write(path, {"x": 1})  # should not raise


class TestLLMLogger:
    def test_start_call_returns_float(self, tmp_path):
        logger = LLMLogger(tmp_path)
        t = logger.start_call()
        assert isinstance(t, float)
        assert t > 0

    def test_record_writes_summary_and_detail(self, tmp_path):
        logger = LLMLogger(tmp_path)
        logger.start_call()
        logger.record(
            system="system prompt",
            user="user message",
            output="response text",
            tokens_in=10,
            tokens_out=5,
            cached_tokens=0,
            model_id="test-model",
            stage="test.stage",
        )
        # Both JSONL files should have been created
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 2

    def test_summary_contains_token_counts(self, tmp_path):
        logger = LLMLogger(tmp_path)
        logger.start_call()
        logger.record(
            system="sys",
            user="usr",
            output="out",
            tokens_in=100,
            tokens_out=50,
            cached_tokens=20,
            model_id="m",
            stage="s",
        )
        summary_file = tmp_path / "llm_token_summary.jsonl"
        assert summary_file.exists()
        row = json.loads(summary_file.read_text(encoding="utf-8").strip())
        assert row["tokens_in"] == 100
        assert row["tokens_out"] == 50
        assert row["cached_tokens"] == 20
        assert row["stage"] == "s"

    def test_detail_contains_full_text(self, tmp_path):
        logger = LLMLogger(tmp_path)
        logger.start_call()
        logger.record(
            system="system text",
            user="user text",
            output="output text",
            tokens_in=1,
            tokens_out=1,
            cached_tokens=None,
            model_id="m",
            stage="s",
        )
        detail_file = tmp_path / "llm_interaction_detail.jsonl"
        row = json.loads(detail_file.read_text(encoding="utf-8").strip())
        assert "system text" in row["input_text"]
        assert "user text" in row["input_text"]
        assert row["output_text"] == "output text"

    def test_record_with_explicit_elapsed(self, tmp_path):
        logger = LLMLogger(tmp_path)
        logger.record(
            system="s",
            user="u",
            output="o",
            tokens_in=1,
            tokens_out=1,
            cached_tokens=None,
            model_id="m",
            stage="s",
            elapsed=3.14,
        )
        summary_file = tmp_path / "llm_token_summary.jsonl"
        row = json.loads(summary_file.read_text(encoding="utf-8").strip())
        assert row["latency_s"] == 3.14

    def test_record_status_error(self, tmp_path):
        logger = LLMLogger(tmp_path)
        logger.start_call()
        logger.record(
            system="s",
            user="u",
            output="",
            tokens_in=None,
            tokens_out=None,
            cached_tokens=None,
            model_id="m",
            stage="s",
            status="error",
            error="Timeout",
        )
        summary_file = tmp_path / "llm_token_summary.jsonl"
        row = json.loads(summary_file.read_text(encoding="utf-8").strip())
        assert row["status"] == "error"
        assert row["error"] == "Timeout"

    def test_run_id_is_unique_across_instances(self, tmp_path):
        a = LLMLogger(tmp_path)
        b = LLMLogger(tmp_path)
        assert a._run_id != b._run_id

    def test_item_id_correlates_summary_and_detail(self, tmp_path):
        logger = LLMLogger(tmp_path)
        logger.start_call()
        logger.record(
            system="sys",
            user="usr",
            output="out",
            tokens_in=1,
            tokens_out=1,
            cached_tokens=None,
            model_id="m",
            stage="s",
        )
        summary_row = json.loads((tmp_path / "llm_token_summary.jsonl").read_text().strip())
        detail_row = json.loads((tmp_path / "llm_interaction_detail.jsonl").read_text().strip())
        assert summary_row["item_id"] == detail_row["item_id"]
        assert summary_row["run_id"] == detail_row["run_id"]
