"""Unit tests for src/stages/_utils.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.stages._utils import CHARS_INVALID, SYSTEM_PAGES, write_atomic


class TestWriteAtomic:
    def test_creates_file_with_content(self, tmp_path):
        dest = tmp_path / "output.md"
        result = write_atomic(dest, "# Hello\n\nContent.")
        assert result is True
        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == "# Hello\n\nContent."

    def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "deep" / "nested" / "file.md"
        write_atomic(dest, "content")
        assert dest.exists()

    def test_skip_if_exists_returns_false_when_file_present(self, tmp_path):
        dest = tmp_path / "existing.md"
        dest.write_text("original", encoding="utf-8")
        result = write_atomic(dest, "new content", skip_if_exists=True)
        assert result is False
        # Original should be unchanged
        assert dest.read_text(encoding="utf-8") == "original"

    def test_skip_if_exists_false_overwrites(self, tmp_path):
        dest = tmp_path / "overwrite.md"
        dest.write_text("original", encoding="utf-8")
        result = write_atomic(dest, "new content", skip_if_exists=False)
        assert result is True
        assert dest.read_text(encoding="utf-8") == "new content"

    def test_no_temp_file_left_on_success(self, tmp_path):
        dest = tmp_path / "file.md"
        write_atomic(dest, "content")
        tmp_files = [f for f in tmp_path.iterdir() if "._tmp_" in f.name]
        assert len(tmp_files) == 0

    def test_cleans_up_temp_on_failure(self, tmp_path):
        dest = tmp_path / "file.md"
        # Make the replace fail by making the destination a directory
        dest.mkdir()
        with pytest.raises(Exception):
            write_atomic(dest, "content")
        # No temp file should remain
        tmp_files = [f for f in tmp_path.iterdir() if "._tmp_" in f.name]
        assert len(tmp_files) == 0

    def test_unicode_content(self, tmp_path):
        dest = tmp_path / "unicode.md"
        content = "# Título\n\nConteúdo com acentos: ção, ã, é."
        write_atomic(dest, content)
        assert dest.read_text(encoding="utf-8") == content

    def test_returns_true_when_file_doesnt_exist_with_skip(self, tmp_path):
        dest = tmp_path / "new_file.md"
        result = write_atomic(dest, "content", skip_if_exists=True)
        assert result is True
        assert dest.exists()


class TestSafeSlug:
    def test_lowercases(self):
        from src.stages._utils import _safe_slug
        # spaces are preserved (not in CHARS_INVALID), but letters are lowercased
        result = _safe_slug("Hello World")
        assert result == result.lower()
        assert "hello" in result
        assert "world" in result

    def test_replaces_invalid_chars(self):
        from src.stages._utils import _safe_slug
        result = _safe_slug("Finance/HR")
        assert "/" not in result
        assert result == "finance-hr"

    def test_collapses_multiple_dashes(self):
        from src.stages._utils import _safe_slug
        result = _safe_slug("word---another")
        assert "--" not in result

    def test_strips_leading_trailing_dashes(self):
        from src.stages._utils import _safe_slug
        result = _safe_slug("  - hello - ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_empty_returns_page(self):
        from src.stages._utils import _safe_slug
        assert _safe_slug("") == "page"


class TestConstants:
    def test_chars_invalid_is_frozenset(self):
        assert isinstance(CHARS_INVALID, frozenset)

    def test_chars_invalid_contains_path_separators(self):
        assert "/" in CHARS_INVALID
        assert "\\" in CHARS_INVALID

    def test_chars_invalid_contains_special_chars(self):
        assert "*" in CHARS_INVALID
        assert "?" in CHARS_INVALID
        assert "<" in CHARS_INVALID
        assert ">" in CHARS_INVALID
        assert "|" in CHARS_INVALID
        assert '"' in CHARS_INVALID

    def test_system_pages_is_frozenset(self):
        assert isinstance(SYSTEM_PAGES, frozenset)

    def test_system_pages_contains_index(self):
        assert "index.md" in SYSTEM_PAGES

    def test_system_pages_contains_log(self):
        assert "log.md" in SYSTEM_PAGES

    def test_system_pages_contains_lint_report(self):
        assert "lint_report.md" in SYSTEM_PAGES
