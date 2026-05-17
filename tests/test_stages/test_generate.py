"""Unit tests for src/stages/generate.py — pure helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig
from src.models.document import Document, DocumentMetadata
from src.stages.generate import (
    _build_frontmatter,
    _raw_id,
    _safe_filename,
    _title_from_draft,
    _truncate,
)


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------


class TestSafeFilename:
    def test_removes_invalid_chars(self):
        result = _safe_filename('report: "Q1/2024"')
        assert ":" not in result
        assert '"' not in result
        assert "/" not in result

    def test_valid_name_unchanged(self):
        assert _safe_filename("my-report_v2") == "my-report_v2"

    def test_strips_leading_dots_and_spaces(self):
        result = _safe_filename(". hidden-file")
        assert not result.startswith(".")
        assert not result.startswith(" ")

    def test_empty_string(self):
        result = _safe_filename("")
        assert result == ""

    def test_all_invalid_chars(self):
        result = _safe_filename(r'\/:*?"<>|')
        assert all(c == "-" for c in result)


# ---------------------------------------------------------------------------
# _title_from_draft
# ---------------------------------------------------------------------------


class TestTitleFromDraft:
    def test_extracts_h1(self):
        draft = "# My Great Title\n\nBody content here."
        assert _title_from_draft(draft) == "MY GREAT TITLE"

    def test_returns_none_if_no_h1(self):
        draft = "## Section Header\n\nNo top-level heading."
        assert _title_from_draft(draft) is None

    def test_strips_whitespace_from_title(self):
        draft = "#   Padded Title   \n\nBody."
        assert _title_from_draft(draft) == "PADDED TITLE"

    def test_uses_first_h1_only(self):
        draft = "# First Title\n\n# Second Title\n\nBody."
        assert _title_from_draft(draft) == "FIRST TITLE"

    def test_empty_string_returns_none(self):
        assert _title_from_draft("") is None

    def test_h1_must_start_with_hash_space(self):
        # "#Title" (no space) should NOT be detected as H1
        draft = "#NotATitle\n\nBody."
        assert _title_from_draft(draft) is None


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_not_truncated(self):
        text, truncated = _truncate("hello", 100)
        assert text == "hello"
        assert truncated is False

    def test_long_text_is_truncated(self):
        text, truncated = _truncate("a" * 200, 100)
        assert len(text) == 100
        assert truncated is True

    def test_exact_length_not_truncated(self):
        text, truncated = _truncate("a" * 100, 100)
        assert len(text) == 100
        assert truncated is False

    def test_empty_string_not_truncated(self):
        text, truncated = _truncate("", 10)
        assert text == ""
        assert truncated is False


# ---------------------------------------------------------------------------
# _raw_id
# ---------------------------------------------------------------------------


class TestRawId:
    def test_appends_raw_suffix(self):
        assert _raw_id("abc123") == "abc123_raw"

    def test_empty_id(self):
        assert _raw_id("") == "_raw"


# ---------------------------------------------------------------------------
# _build_frontmatter
# ---------------------------------------------------------------------------


class TestBuildFrontmatter:
    def _make_doc(self, extra: dict | None = None) -> Document:
        return Document(
            metadata=DocumentMetadata(
                id="550e8400-e29b-41d4-a716-446655440000",
                title="My Document",
                object_type="article",
                status="active",
                extra=extra or {"content_sha256": "abc123", "source_filename": "doc.md"},
            ),
            content="Body text.",
        )

    def _make_obj_cfg(self, tmp_path: Path) -> ObjectTypeConfig:
        g = tmp_path / "gen.md"
        g.write_text("prompt")
        e = tmp_path / "eval.md"
        e.write_text("prompt")
        return ObjectTypeConfig(
            name="Article",
            slug="article",
            wiki_subdir="articles",
            prompt_generate=g,
            prompt_evaluate=e,
        )

    def test_contains_id(self, tmp_path):
        doc = self._make_doc()
        obj_cfg = self._make_obj_cfg(tmp_path)
        fm = _build_frontmatter(doc, obj_cfg, "2024-01-01T00:00:00+00:00")
        assert "550e8400-e29b-41d4-a716-446655440000" in fm

    def test_contains_title(self, tmp_path):
        doc = self._make_doc()
        obj_cfg = self._make_obj_cfg(tmp_path)
        fm = _build_frontmatter(doc, obj_cfg, "2024-01-01T00:00:00+00:00")
        assert "My Document" in fm

    def test_contains_object_type(self, tmp_path):
        doc = self._make_doc()
        obj_cfg = self._make_obj_cfg(tmp_path)
        fm = _build_frontmatter(doc, obj_cfg, "2024-01-01T00:00:00+00:00")
        assert "object_type: article" in fm

    def test_contains_status(self, tmp_path):
        doc = self._make_doc()
        obj_cfg = self._make_obj_cfg(tmp_path)
        fm = _build_frontmatter(doc, obj_cfg, "2024-01-01T00:00:00+00:00")
        assert "status: active" in fm

    def test_contains_frontmatter_delimiters(self, tmp_path):
        doc = self._make_doc()
        obj_cfg = self._make_obj_cfg(tmp_path)
        fm = _build_frontmatter(doc, obj_cfg, "2024-01-01T00:00:00+00:00")
        lines = fm.splitlines()
        assert lines[0] == "---"
        assert "---" in lines[1:]

    def test_includes_configured_frontmatter_fields(self, tmp_path):
        doc = self._make_doc(extra={"author": "Alice", "content_sha256": "x", "source_filename": "doc.md"})
        g = tmp_path / "gen.md"
        g.write_text("prompt")
        e = tmp_path / "eval.md"
        e.write_text("prompt")
        obj_cfg = ObjectTypeConfig(
            name="Article",
            slug="article",
            wiki_subdir="articles",
            prompt_generate=g,
            prompt_evaluate=e,
            frontmatter_fields=["author"],
        )
        fm = _build_frontmatter(doc, obj_cfg, "2024-01-01T00:00:00+00:00")
        assert "author: Alice" in fm

    def test_source_raw_link_present(self, tmp_path):
        doc = self._make_doc()
        obj_cfg = self._make_obj_cfg(tmp_path)
        fm = _build_frontmatter(doc, obj_cfg, "2024-01-01T00:00:00+00:00")
        assert "source_raw:" in fm
        assert "_raw" in fm


# ---------------------------------------------------------------------------
# _write_raw_page
# ---------------------------------------------------------------------------


class TestWriteRawPage:
    def _make_doc(self, tmp_path: Path) -> Document:
        return Document(
            metadata=DocumentMetadata(
                id="test-uuid-001",
                title="My Source Doc",
                object_type="article",
                extra={"content_sha256": "deadbeef", "source_filename": "source.md"},
            ),
            content="# My Source Doc\n\nOriginal content here.\n",
        )

    def _make_obj_cfg(self, tmp_path: Path) -> "ObjectTypeConfig":
        from src.models.config import ObjectTypeConfig
        g = tmp_path / "gen.md"; g.write_text("prompt")
        e = tmp_path / "eval.md"; e.write_text("prompt")
        return ObjectTypeConfig(
            name="Article", slug="article", wiki_subdir="articles",
            prompt_generate=g, prompt_evaluate=e,
        )

    def test_creates_raw_file(self, tmp_path):
        from src.stages.generate import _write_raw_page
        doc = self._make_doc(tmp_path)
        obj_cfg = self._make_obj_cfg(tmp_path)
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        path = _write_raw_page(doc, obj_cfg, wiki_dir, "2024-01-01T00:00:00+00:00")
        assert path.exists()
        assert path.name.endswith("_raw.md")

    def test_raw_file_contains_title(self, tmp_path):
        from src.stages.generate import _write_raw_page
        doc = self._make_doc(tmp_path)
        obj_cfg = self._make_obj_cfg(tmp_path)
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        path = _write_raw_page(doc, obj_cfg, wiki_dir, "2024-01-01T00:00:00+00:00")
        content = path.read_text(encoding="utf-8")
        assert "My Source Doc" in content
        assert "deadbeef" in content

    def test_raw_file_in_correct_subdir(self, tmp_path):
        from src.stages.generate import _write_raw_page
        doc = self._make_doc(tmp_path)
        obj_cfg = self._make_obj_cfg(tmp_path)
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()

        path = _write_raw_page(doc, obj_cfg, wiki_dir, "2024-01-01T00:00:00+00:00")
        assert "articles" in str(path)
        assert "raw" in str(path)


# ---------------------------------------------------------------------------
# generate_page (async, with mock LLM)
# ---------------------------------------------------------------------------


class TestGeneratePage:
    def _make_cfg(self, tmp_path: Path) -> "WikiConfig":
        from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig
        wiki = tmp_path / "wiki"; wiki.mkdir()
        logs = tmp_path / "logs"; logs.mkdir()
        prompts = tmp_path / "prompts"; prompts.mkdir()
        (prompts / "gen.md").write_text("Generate: {{ document }}")
        (prompts / "eval.md").write_text("Evaluate: {{ draft }}")
        (prompts / "editor.md").write_text("Edit: {{ draft }}")
        (prompts / "lint.md").write_text("Lint")
        (prompts / "consol.md").write_text("Consolidate")
        (prompts / "chat.md").write_text("Chat")
        return WikiConfig(
            wiki_name="Test", wiki_dir=wiki, log_dir=logs,
            llm=LLMConfig(backend="openrouter", model_id="test"),
            objects=[ObjectTypeConfig(
                name="Article", slug="article", wiki_subdir="articles",
                prompt_generate=prompts / "gen.md",
                prompt_evaluate=prompts / "eval.md",
            )],
            prompt_editor=prompts / "editor.md",
            prompt_lint=prompts / "lint.md",
            prompt_consolidate=prompts / "consol.md",
            prompt_chat=prompts / "chat.md",
        )

    def _make_doc(self) -> Document:
        return Document(
            metadata=DocumentMetadata(
                id="gen-test-001",
                title="Test Article",
                object_type="article",
                extra={"content_sha256": "abc", "source_filename": "src.md"},
            ),
            content="Content of the article.",
        )

    @pytest.mark.asyncio
    async def test_generate_page_creates_file(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.stages.generate import generate_page
        from src.models.evaluation import PageEvaluation

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()
        draft = "# Test Article\n\nGenerated content."

        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        with patch("src.stages.generate._generate_draft", AsyncMock(return_value=draft)), \
             patch("src.stages.generate._evaluate_draft",
                   AsyncMock(return_value=PageEvaluation(approved=True))):
            result = await generate_page(doc, cfg, MagicMock(), mock_logger, force=True)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".md"

    @pytest.mark.asyncio
    async def test_generate_page_skips_existing(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.generate import generate_page

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()

        # Pre-create the page with matching id in frontmatter
        articles_dir = cfg.wiki_dir / "articles"
        articles_dir.mkdir(exist_ok=True)
        existing = articles_dir / "test-article.md"
        existing.write_text(
            '---\nid: "gen-test-001"\ntitle: "Test Article"\n---\n# Test\n',
            encoding="utf-8",
        )
        raw_dir = articles_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / "gen-test-001_raw.md").write_text("raw content")

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock()
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await generate_page(doc, cfg, mock_llm, mock_logger, force=False)
        # Should return existing without calling LLM
        mock_llm.call.assert_not_awaited()
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_page_returns_none_on_error_skip(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.stages.generate import generate_page

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()

        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        with patch("src.stages.generate._generate_draft",
                   AsyncMock(side_effect=RuntimeError("LLM failed"))):
            result = await generate_page(doc, cfg, MagicMock(), mock_logger, force=True)

        assert result is None  # on_llm_error defaults to "skip"


# ---------------------------------------------------------------------------
# _generate_draft and _evaluate_draft (internal async functions)
# ---------------------------------------------------------------------------


class TestGenerateDraft:
    def _make_cfg(self, tmp_path: Path) -> "WikiConfig":
        wiki = tmp_path / "wiki"; wiki.mkdir()
        logs = tmp_path / "logs"; logs.mkdir()
        prompts = tmp_path / "prompts"; prompts.mkdir()
        (prompts / "gen.md").write_text("Generate: {{ document }}")
        (prompts / "eval.md").write_text("Evaluate: {{ draft }}")
        (prompts / "editor.md").write_text("Edit")
        (prompts / "lint.md").write_text("Lint")
        (prompts / "consol.md").write_text("Consol")
        (prompts / "chat.md").write_text("Chat")
        return WikiConfig(
            wiki_name="Test", wiki_dir=wiki, log_dir=logs,
            llm=LLMConfig(backend="openrouter", model_id="test"),
            objects=[ObjectTypeConfig(
                name="Article", slug="article", wiki_subdir="articles",
                prompt_generate=prompts / "gen.md",
                prompt_evaluate=prompts / "eval.md",
            )],
            prompt_editor=prompts / "editor.md",
            prompt_lint=prompts / "lint.md",
            prompt_consolidate=prompts / "consol.md",
            prompt_chat=prompts / "chat.md",
        )

    def _make_doc(self) -> Document:
        return Document(
            metadata=DocumentMetadata(
                id="draft-test-001",
                title="Draft Test",
                object_type="article",
                extra={"content_sha256": "abc", "source_filename": "src.md"},
            ),
            content="Some article content.",
        )

    @pytest.mark.asyncio
    async def test_generate_draft_calls_llm(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.llm.base import LLMResponse
        from src.stages.generate import _generate_draft

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()
        obj_cfg = cfg.objects[0]

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=LLMResponse(
            text="# Draft Article\n\nContent.", tokens_in=10, tokens_out=20,
            cached_tokens=None, model_id="test", attempts=1,
        ))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await _generate_draft(doc, obj_cfg, mock_llm, cfg, mock_logger)
        assert result == "# Draft Article\n\nContent."
        mock_llm.call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_draft_returns_approval(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.models.evaluation import PageEvaluation
        from src.stages.generate import _evaluate_draft

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()
        obj_cfg = cfg.objects[0]

        mock_eval = PageEvaluation(approved=True, problems=[], suggestions=[])
        mock_llm = MagicMock()
        mock_llm.call_structured = AsyncMock(return_value=mock_eval)
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await _evaluate_draft("# Draft\n\nContent.", doc, obj_cfg, mock_llm, cfg, mock_logger)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_evaluate_draft_auto_approves_on_error(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.generate import _evaluate_draft

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()
        obj_cfg = cfg.objects[0]

        mock_llm = MagicMock()
        mock_llm.call_structured = AsyncMock(side_effect=RuntimeError("evaluator failed"))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await _evaluate_draft("# Draft\n\nContent.", doc, obj_cfg, mock_llm, cfg, mock_logger)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_edit_draft_calls_llm(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.llm.base import LLMResponse
        from src.models.evaluation import PageEvaluation
        from src.stages.generate import _edit_draft

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()

        evaluation = PageEvaluation(
            approved=False, problems=["Too short"], suggestions=["Add more detail"]
        )
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=LLMResponse(
            text="# Improved Draft\n\nMore content.", tokens_in=15, tokens_out=25,
            cached_tokens=None, model_id="test", attempts=1,
        ))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await _edit_draft("# Old Draft\n\nShort.", evaluation, doc, cfg, mock_llm, mock_logger)
        assert result == "# Improved Draft\n\nMore content."
        mock_llm.call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_page_with_edit_rounds(self, tmp_path):
        """Test generate_page with evaluation rejection and editing."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.models.evaluation import PageEvaluation
        from src.stages.generate import generate_page

        cfg = self._make_cfg(tmp_path)
        doc = self._make_doc()
        initial_draft = "# Draft Article\n\nShort content."
        edited_draft = "# Draft Article\n\nImproved content with more detail."

        reject = PageEvaluation(approved=False, problems=["Too short"], suggestions=["Expand"])
        approve = PageEvaluation(approved=True)

        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        with patch("src.stages.generate._generate_draft", AsyncMock(return_value=initial_draft)), \
             patch("src.stages.generate._evaluate_draft",
                   AsyncMock(side_effect=[reject, approve])), \
             patch("src.stages.generate._edit_draft",
                   AsyncMock(return_value=edited_draft)):
            result = await generate_page(doc, cfg, MagicMock(), mock_logger, force=True)

        assert result is not None
        assert result.exists()
        assert "Improved content" in result.read_text(encoding="utf-8")
