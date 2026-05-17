"""Unit tests for src/stages/lint.py — static analysis functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stages.lint import _extract_wikilinks, _load_pages, static_analysis


class TestExtractWikilinks:
    def test_empty_text(self):
        assert _extract_wikilinks("") == set()

    def test_single_link(self):
        result = _extract_wikilinks("See [[my-page]] for details.")
        assert result == {"my-page"}

    def test_multiple_links(self):
        result = _extract_wikilinks("Links: [[page-a]], [[page-b]], [[page-c]].")
        assert result == {"page-a", "page-b", "page-c"}

    def test_duplicate_links_deduplicated(self):
        result = _extract_wikilinks("[[link]] and [[link]] again.")
        assert result == {"link"}

    def test_no_wikilinks(self):
        result = _extract_wikilinks("Plain text with [regular](links.md) but no wikilinks.")
        assert result == set()

    def test_wikilink_with_spaces(self):
        result = _extract_wikilinks("[[My Page Title]]")
        assert "My Page Title" in result


class TestLoadPages:
    def test_loads_md_files(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page-a.md").write_text("# Page A\n\nContent.", encoding="utf-8")
        (wiki_dir / "page-b.md").write_text("# Page B\n\nContent.", encoding="utf-8")

        pages = _load_pages(wiki_dir)
        assert "page-a" in pages
        assert "page-b" in pages

    def test_excludes_system_pages(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "index.md").write_text("# Index\n")
        (wiki_dir / "log.md").write_text("# Log\n")
        (wiki_dir / "lint_report.md").write_text("# Lint\n")
        (wiki_dir / "real-page.md").write_text("# Real Page\n")

        pages = _load_pages(wiki_dir)
        assert "index" not in pages
        assert "log" not in pages
        assert "lint_report" not in pages
        assert "real-page" in pages

    def test_excludes_lint_prefixed_files(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "lint_something.md").write_text("# Lint\n")
        (wiki_dir / "real.md").write_text("# Real\n")

        pages = _load_pages(wiki_dir)
        assert "lint_something" not in pages
        assert "real" in pages

    def test_recurses_into_subdirs(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        subdir = wiki_dir / "articles"
        subdir.mkdir(parents=True)
        (subdir / "deep-page.md").write_text("# Deep Page\n")

        pages = _load_pages(wiki_dir)
        assert "deep-page" in pages

    def test_returns_file_content(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text("# My Content\n\nBody text.", encoding="utf-8")

        pages = _load_pages(wiki_dir)
        assert "# My Content" in pages["page"]


class TestStaticAnalysis:
    def test_nonexistent_wiki_dir(self, tmp_path):
        result = static_analysis(tmp_path / "nonexistent")
        assert result["orphans"] == []
        assert result["broken_links"] == []
        assert result["stats"]["total_paginas"] == 0

    def test_single_page_is_orphan(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "lonely-page.md").write_text("# Lonely\n\nNo links here.", encoding="utf-8")

        result = static_analysis(wiki_dir)
        assert "lonely-page" in result["orphans"]

    def test_linked_page_is_not_orphan(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "source.md").write_text("# Source\n\nSee [[target]].", encoding="utf-8")
        (wiki_dir / "target.md").write_text("# Target\n\nContent.", encoding="utf-8")
        # Create index.md to link source so it's not an orphan
        (wiki_dir / "index.md").write_text("[[source]] [[target]]", encoding="utf-8")

        result = static_analysis(wiki_dir)
        assert "target" not in result["orphans"]

    def test_detects_broken_link(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "source.md").write_text(
            "# Source\n\nSee [[nonexistent-page]].", encoding="utf-8"
        )

        result = static_analysis(wiki_dir)
        broken = result["broken_links"]
        assert any(b["source"] == "source" and b["target"] == "nonexistent-page" for b in broken)

    def test_no_broken_links_when_target_exists(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "source.md").write_text("# Source\n\nSee [[target]].", encoding="utf-8")
        (wiki_dir / "target.md").write_text("# Target\n\nContent.", encoding="utf-8")

        result = static_analysis(wiki_dir)
        assert not any(b["target"] == "target" for b in result["broken_links"])

    def test_stats_count_pages_and_links(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page-a.md").write_text("# A\n\n[[page-b]] [[page-c]]", encoding="utf-8")
        (wiki_dir / "page-b.md").write_text("# B\n\nContent.", encoding="utf-8")
        (wiki_dir / "page-c.md").write_text("# C\n\nContent.", encoding="utf-8")

        result = static_analysis(wiki_dir)
        assert result["stats"]["total_pages"] == 3
        assert result["stats"]["total_links"] >= 2

    def test_index_md_provides_inbound_links(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "referenced.md").write_text("# Ref\n\nContent.", encoding="utf-8")
        (wiki_dir / "index.md").write_text("[[referenced]]", encoding="utf-8")

        result = static_analysis(wiki_dir)
        assert "referenced" not in result["orphans"]

    def test_self_link_excluded_from_index_links(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text("# Page\n\nLink to [[index]].", encoding="utf-8")

        result = static_analysis(wiki_dir)
        # "index" links should be stripped from outbound (see code: - {"index"})
        assert not any(b["target"] == "index" for b in result["broken_links"])


# ---------------------------------------------------------------------------
# run_lint (async, with mock LLM)
# ---------------------------------------------------------------------------


class TestRunLint:
    def _make_cfg(self, tmp_path: Path):
        from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig
        wiki = tmp_path / "wiki"; wiki.mkdir()
        logs = tmp_path / "logs"; logs.mkdir()
        prompts = tmp_path / "prompts"; prompts.mkdir()
        for fn in ["gen.md", "eval.md", "editor.md", "lint.md", "consol.md", "chat.md"]:
            (prompts / fn).write_text("prompt text")
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

    @pytest.mark.asyncio
    async def test_run_lint_writes_report(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.llm.base import LLMResponse
        from src.stages.lint import run_lint

        cfg = self._make_cfg(tmp_path)
        (cfg.wiki_dir / "page-a.md").write_text("# A\n\n[[page-b]]", encoding="utf-8")
        (cfg.wiki_dir / "page-b.md").write_text("# B\n\nContent.", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=LLMResponse(
            text="No major issues.", tokens_in=5, tokens_out=5,
            cached_tokens=None, model_id="test", attempts=1,
        ))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await run_lint(cfg, mock_llm, mock_logger)

        assert (cfg.wiki_dir / "lint_report.md").exists()
        from src.models.evaluation import RepairState
        assert isinstance(result, RepairState)

    @pytest.mark.asyncio
    async def test_run_lint_returns_orphans(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.llm.base import LLMResponse
        from src.stages.lint import run_lint

        cfg = self._make_cfg(tmp_path)
        (cfg.wiki_dir / "lonely.md").write_text("# Lonely\n\nNo links.", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=LLMResponse(
            text="Found orphan.", tokens_in=3, tokens_out=2,
            cached_tokens=None, model_id="test", attempts=1,
        ))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await run_lint(cfg, mock_llm, mock_logger)
        assert "lonely" in result.orphans

    @pytest.mark.asyncio
    async def test_run_lint_handles_llm_error(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.lint import run_lint

        cfg = self._make_cfg(tmp_path)
        (cfg.wiki_dir / "page.md").write_text("# Page\n\nContent.", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        # Should not raise — handles error gracefully
        result = await run_lint(cfg, mock_llm, mock_logger)
        assert result is not None
