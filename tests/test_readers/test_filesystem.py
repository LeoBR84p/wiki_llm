"""Unit tests for src/readers/filesystem.py."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig
from src.readers.filesystem import (
    FilesystemReader,
    _extract_md,
    _humanize,
    _read_file_sync,
    _slugify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wiki_config(tmp_path: Path, content_dir: Path | None = None) -> WikiConfig:
    """Build a minimal WikiConfig pointing at tmp directories."""
    prompts = tmp_path / "prompts"
    prompts.mkdir(exist_ok=True)
    for name in ["gen.md", "eval.md", "editor.md", "lint.md", "consol.md", "chat.md"]:
        (prompts / name).write_text("prompt")

    return WikiConfig(
        wiki_name="Test",
        wiki_dir=tmp_path / "wiki",
        log_dir=tmp_path / "logs",
        content_dir=content_dir or (tmp_path / "content"),
        llm=LLMConfig(backend="openrouter", model_id="test"),
        objects=[
            ObjectTypeConfig(
                name="Article",
                slug="article",
                wiki_subdir="articles",
                prompt_generate=prompts / "gen.md",
                prompt_evaluate=prompts / "eval.md",
            )
        ],
        prompt_editor=prompts / "editor.md",
        prompt_lint=prompts / "lint.md",
        prompt_consolidate=prompts / "consol.md",
        prompt_chat=prompts / "chat.md",
    )


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_lowercase(self):
        assert _slugify("Hello World") == "hello_world"

    def test_special_chars_replaced(self):
        # Non-alphanumeric sequences replaced with _, trailing _ stripped
        assert _slugify("Hello-World!") == "hello_world"

    def test_strips_leading_trailing_underscores(self):
        result = _slugify("  hello  ")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_already_slugified(self):
        assert _slugify("my_slug") == "my_slug"

    def test_numbers_preserved(self):
        assert "2024" in _slugify("Report 2024")


# ---------------------------------------------------------------------------
# _humanize
# ---------------------------------------------------------------------------


class TestHumanize:
    def test_basic(self):
        assert _humanize("my_document") == "My Document"

    def test_dashes(self):
        assert _humanize("my-document") == "My Document"

    def test_mixed(self):
        assert _humanize("my-doc_v2") == "My Doc V2"

    def test_single_word(self):
        assert _humanize("report") == "Report"

    def test_already_titled(self):
        result = _humanize("Annual Report")
        assert result == "Annual Report"


# ---------------------------------------------------------------------------
# _extract_md
# ---------------------------------------------------------------------------


class TestExtractMd:
    def test_parses_frontmatter_title(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "test.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "---\ntitle: My Test Doc\nobject_type: article\n---\n# Body\n\nContent here."
        path.write_text(raw, encoding="utf-8")

        doc = _extract_md(raw, path, cfg, "article")
        assert doc.metadata.title == "My Test Doc"
        assert doc.metadata.object_type == "article"

    def test_falls_back_to_filename_title(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "my_great_document.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "# Body\n\nNo frontmatter here."
        path.write_text(raw, encoding="utf-8")

        doc = _extract_md(raw, path, cfg, "article")
        assert doc.metadata.title == "My Great Document"

    def test_uuid_is_deterministic(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "doc.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "# Fixed Content\n\nAlways the same."
        path.write_text(raw, encoding="utf-8")

        doc1 = _extract_md(raw, path, cfg, "article")
        doc2 = _extract_md(raw, path, cfg, "article")
        assert doc1.metadata.id == doc2.metadata.id

    def test_uuid_changes_with_content(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "doc.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        raw1 = "# Content A\n\nFirst version."
        raw2 = "# Content B\n\nSecond version."
        path.write_text(raw1, encoding="utf-8")
        doc1 = _extract_md(raw1, path, cfg, "article")
        path.write_text(raw2, encoding="utf-8")
        doc2 = _extract_md(raw2, path, cfg, "article")
        assert doc1.metadata.id != doc2.metadata.id

    def test_status_extracted_from_frontmatter(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "active.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "---\ntitle: Active Doc\nstatus: active\n---\n# Body"
        path.write_text(raw, encoding="utf-8")
        doc = _extract_md(raw, path, cfg, "article")
        assert doc.metadata.status == "active"

    def test_content_sha256_in_extra(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "doc.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "# Content\n\nSome text."
        path.write_text(raw, encoding="utf-8")
        doc = _extract_md(raw, path, cfg, "article")
        assert "content_sha256" in doc.metadata.extra
        assert len(doc.metadata.extra["content_sha256"]) == 64  # SHA-256 hex

    def test_source_filename_in_extra(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "my_file.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "# Doc\n\nContent."
        path.write_text(raw, encoding="utf-8")
        doc = _extract_md(raw, path, cfg, "article")
        assert doc.metadata.extra["source_filename"] == "my_file.md"

    def test_title_from_frontmatter(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        path = tmp_path / "content" / "pt_doc.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "---\ntitle: Meu Documento\n---\n# Body"
        path.write_text(raw, encoding="utf-8")
        doc = _extract_md(raw, path, cfg, "article")
        assert doc.metadata.title == "Meu Documento"


# ---------------------------------------------------------------------------
# _read_file_sync
# ---------------------------------------------------------------------------


class TestReadFileSync:
    def test_reads_md_file(self, tmp_path):
        cfg = _make_wiki_config(tmp_path)
        content_dir = tmp_path / "content"
        content_dir.mkdir(exist_ok=True)
        cfg = _make_wiki_config(tmp_path, content_dir)

        path = content_dir / "test.md"
        path.write_text("---\ntitle: Test\n---\n# Body\n\nContent.", encoding="utf-8")
        doc = _read_file_sync(path, cfg)
        assert doc is not None
        assert doc.metadata.title == "Test"

    def test_reads_txt_file(self, tmp_path):
        cfg = _make_wiki_config(tmp_path, tmp_path / "content")
        content_dir = tmp_path / "content"
        content_dir.mkdir(exist_ok=True)
        path = content_dir / "note.txt"
        path.write_text("Plain text content here.", encoding="utf-8")
        doc = _read_file_sync(path, cfg)
        assert doc is not None
        assert doc.content == "Plain text content here."

    def test_skips_pdf_without_reader(self, tmp_path):
        cfg = _make_wiki_config(tmp_path, tmp_path / "content")
        content_dir = tmp_path / "content"
        content_dir.mkdir(exist_ok=True)
        path = content_dir / "doc.pdf"
        path.write_bytes(b"%PDF-1.4 fake content")
        doc = _read_file_sync(path, cfg)
        assert doc is None

    def test_object_type_from_parent_subdir(self, tmp_path):
        """Files inside a known-slug subdirectory inherit that slug."""
        content_dir = tmp_path / "content"
        article_dir = content_dir / "article"
        article_dir.mkdir(parents=True)
        cfg = _make_wiki_config(tmp_path, content_dir)

        path = article_dir / "subdir_doc.md"
        path.write_text("# Article Doc\n\nContent.", encoding="utf-8")
        doc = _read_file_sync(path, cfg)
        assert doc is not None
        assert doc.metadata.object_type == "article"

    def test_object_type_falls_back_to_first(self, tmp_path):
        """Files in unrecognized subdirectory fall back to first object slug."""
        content_dir = tmp_path / "content"
        unknown_dir = content_dir / "unknown_subdir"
        unknown_dir.mkdir(parents=True)
        cfg = _make_wiki_config(tmp_path, content_dir)

        path = unknown_dir / "unknown.md"
        path.write_text("# Unknown\n\nContent.", encoding="utf-8")
        doc = _read_file_sync(path, cfg)
        assert doc is not None
        assert doc.metadata.object_type == "article"  # first slug fallback

    def test_reads_docx_file_with_markitdown(self, tmp_path):
        """docx processing delegates to markitdown and returns a Document."""
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        path = content_dir / "report.docx"
        path.write_bytes(b"fake docx content")

        mock_instance = MagicMock()
        mock_instance.convert.return_value = MagicMock(text_content="# Report\n\nDocx content.")
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("markitdown.MarkItDown", mock_cls):
            doc = _read_file_sync(path, cfg)

        assert doc is not None
        assert "Docx content" in doc.content

    def test_reads_pdf_with_configured_reader(self, tmp_path):
        """pdf with a configured pdf_reader calls extract_text and returns Document."""
        from unittest.mock import MagicMock

        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        # Attach a fake pdf_reader
        mock_pdf_reader = MagicMock()
        mock_pdf_reader.extract_text.return_value = "# PDF Title\n\nExtracted content."
        cfg = cfg.model_copy(update={"pdf_reader": mock_pdf_reader})

        path = content_dir / "doc.pdf"
        path.write_bytes(b"%PDF-1.4 fake")

        doc = _read_file_sync(path, cfg)
        assert doc is not None
        mock_pdf_reader.extract_text.assert_called_once_with(path)

    def test_pdf_with_empty_text_raises_value_error(self, tmp_path):
        """pdf_reader returning empty text raises ValueError."""
        from unittest.mock import MagicMock

        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        mock_pdf_reader = MagicMock()
        mock_pdf_reader.extract_text.return_value = "   "  # whitespace only
        cfg = cfg.model_copy(update={"pdf_reader": mock_pdf_reader})

        path = content_dir / "empty.pdf"
        path.write_bytes(b"%PDF-1.4 fake")

        with pytest.raises(ValueError, match="no extractable text"):
            _read_file_sync(path, cfg)

    def test_docx_with_empty_text_raises_value_error(self, tmp_path):
        """docx returning empty content raises ValueError (propagated)."""
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        path = content_dir / "empty.docx"
        path.write_bytes(b"fake docx")

        mock_instance = MagicMock()
        mock_instance.convert.return_value = MagicMock(text_content="   ")
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("markitdown.MarkItDown", mock_cls):
            with pytest.raises(ValueError):
                _read_file_sync(path, cfg)

    def test_general_exception_is_reraised(self, tmp_path):
        """Unexpected exceptions from file reading are re-raised."""
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        path = content_dir / "broken.docx"
        path.write_bytes(b"fake docx")

        mock_instance = MagicMock()
        mock_instance.convert.side_effect = OSError("disk error")
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("markitdown.MarkItDown", mock_cls):
            with pytest.raises(OSError):
                _read_file_sync(path, cfg)


# ---------------------------------------------------------------------------
# FilesystemReader
# ---------------------------------------------------------------------------


class TestFilesystemReader:
    @pytest.mark.asyncio
    async def test_empty_content_dir(self, tmp_path):
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)
        reader = FilesystemReader(cfg)
        docs = await reader.read_all()
        assert docs == []

    @pytest.mark.asyncio
    async def test_nonexistent_content_dir(self, tmp_path):
        cfg = _make_wiki_config(tmp_path, tmp_path / "nonexistent")
        reader = FilesystemReader(cfg)
        docs = await reader.read_all()
        assert docs == []

    @pytest.mark.asyncio
    async def test_reads_multiple_md_files(self, tmp_path):
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        for i in range(3):
            (content_dir / f"doc_{i}.md").write_text(
                f"---\ntitle: Doc {i}\n---\n# Doc {i}\n\nContent {i}.",
                encoding="utf-8",
            )
        reader = FilesystemReader(cfg)
        docs = await reader.read_all()
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_ignores_unsupported_extensions(self, tmp_path):
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        (content_dir / "doc.md").write_text("# Doc\n\nContent.", encoding="utf-8")
        (content_dir / "image.png").write_bytes(b"\x89PNG fake")
        (content_dir / "data.csv").write_text("a,b,c", encoding="utf-8")

        reader = FilesystemReader(cfg)
        docs = await reader.read_all()
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_errors_are_skipped_not_raised(self, tmp_path):
        """A file that raises during reading should not abort the whole batch."""
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cfg = _make_wiki_config(tmp_path, content_dir)

        # A valid document
        (content_dir / "good.md").write_text("# Good\n\nContent.", encoding="utf-8")
        # An unreadable file - simulate by writing binary garbage with .md ext
        bad_file = content_dir / "bad.md"
        bad_file.write_bytes(b"\xff\xfe invalid utf-8 \x00\x01")

        reader = FilesystemReader(cfg)
        # Should not raise; reads what it can
        docs = await reader.read_all()
        # At least the good one should be read
        assert any(d.metadata.title == "Good" for d in docs)
