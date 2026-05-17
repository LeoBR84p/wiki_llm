"""Unit tests for src/stages/consolidate.py — pure functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.stages.consolidate import (
    _add_aliases,
    _collect_pages,
    _execute_merge,
    _replace_wikilinks,
    _safe_slug,
)


# ---------------------------------------------------------------------------
# _safe_slug
# ---------------------------------------------------------------------------


class TestSafeSlug:
    def test_lowercases(self):
        # spaces are preserved, only CHARS_INVALID chars become dashes
        assert _safe_slug("Hello World") == "hello world"

    def test_replaces_invalid_chars(self):
        result = _safe_slug("Finance/HR")
        assert "/" not in result

    def test_collapses_multiple_dashes(self):
        result = _safe_slug("word---another")
        assert "--" not in result

    def test_strips_leading_trailing_dashes(self):
        result = _safe_slug("  - hello - ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_empty_returns_page(self):
        assert _safe_slug("") == "page"

    def test_colon_becomes_dash(self):
        # colon is in CHARS_INVALID, parens are NOT
        result = _safe_slug("A: B")
        assert ":" not in result


# ---------------------------------------------------------------------------
# _collect_pages
# ---------------------------------------------------------------------------


class TestCollectPages:
    def test_collects_md_files(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "page-a.md").write_text("# Page A\n", encoding="utf-8")
        (subdir / "page-b.md").write_text("# Page B\n", encoding="utf-8")

        pages = _collect_pages(subdir)
        slugs = [p["slug"] for p in pages]
        assert "page-a" in slugs
        assert "page-b" in slugs

    def test_extracts_title_from_frontmatter(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        content = '---\ntitle: "My Article Title"\n---\n# Body\n'
        (subdir / "my-article.md").write_text(content, encoding="utf-8")

        pages = _collect_pages(subdir)
        assert pages[0]["name"] == "My Article Title"

    def test_falls_back_to_stem_when_no_frontmatter(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "plain-article.md").write_text("# Body\n", encoding="utf-8")

        pages = _collect_pages(subdir)
        assert pages[0]["name"] == "plain-article"

    def test_excludes_system_pages(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "index.md").write_text("# Index\n")
        (subdir / "log.md").write_text("# Log\n")
        (subdir / "real.md").write_text("# Real\n")

        pages = _collect_pages(subdir)
        slugs = [p["slug"] for p in pages]
        assert "index" not in slugs
        assert "log" not in slugs
        assert "real" in slugs

    def test_returns_sorted_by_name(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "z-page.md").write_text("# Z\n")
        (subdir / "a-page.md").write_text("# A\n")

        pages = _collect_pages(subdir)
        slugs = [p["slug"] for p in pages]
        assert slugs == sorted(slugs)


# ---------------------------------------------------------------------------
# _replace_wikilinks
# ---------------------------------------------------------------------------


class TestReplaceWikilinks:
    def test_replaces_link_in_file(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        page = wiki_dir / "source.md"
        page.write_text("# Source\n\nSee [[old-page]] for details.", encoding="utf-8")

        count = _replace_wikilinks(wiki_dir, "old-page", "new-page")
        assert count == 1
        assert "[[new-page]]" in page.read_text(encoding="utf-8")
        assert "[[old-page]]" not in page.read_text(encoding="utf-8")

    def test_returns_count_of_modified_files(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        for i in range(3):
            (wiki_dir / f"page-{i}.md").write_text(f"# Page {i}\n\n[[dup-link]]", encoding="utf-8")

        count = _replace_wikilinks(wiki_dir, "dup-link", "canonical-link")
        assert count == 3

    def test_skips_system_pages(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "index.md").write_text("[[old-page]]", encoding="utf-8")
        (wiki_dir / "log.md").write_text("[[old-page]]", encoding="utf-8")
        (wiki_dir / "lint_report.md").write_text("[[old-page]]", encoding="utf-8")

        # System pages are NOT modified
        count = _replace_wikilinks(wiki_dir, "old-page", "new-page")
        assert count == 0

    def test_no_match_returns_zero(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text("# Page\n\n[[other-link]]", encoding="utf-8")

        count = _replace_wikilinks(wiki_dir, "nonexistent", "new-page")
        assert count == 0

    def test_recurses_into_subdirs(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        subdir = wiki_dir / "articles"
        subdir.mkdir(parents=True)
        (subdir / "article.md").write_text("[[old-link]]", encoding="utf-8")

        count = _replace_wikilinks(wiki_dir, "old-link", "new-link")
        assert count == 1


# ---------------------------------------------------------------------------
# _add_aliases
# ---------------------------------------------------------------------------


class TestAddAliases:
    def test_adds_aliases_to_page_without_aliases(self, tmp_path):
        page = tmp_path / "page.md"
        page.write_text('---\ntitle: "My Page"\n---\n# Body\n', encoding="utf-8")

        _add_aliases(page, ["Old Name", "Former Title"])
        content = page.read_text(encoding="utf-8")
        assert "aliases:" in content
        assert "Old Name" in content

    def test_appends_to_existing_aliases(self, tmp_path):
        page = tmp_path / "page.md"
        page.write_text(
            '---\ntitle: "Page"\naliases: ["Existing Alias"]\n---\n# Body\n',
            encoding="utf-8",
        )
        _add_aliases(page, ["New Alias"])
        content = page.read_text(encoding="utf-8")
        assert "Existing Alias" in content
        assert "New Alias" in content

    def test_empty_aliases_does_nothing(self, tmp_path):
        page = tmp_path / "page.md"
        original = '---\ntitle: "Page"\n---\n# Body\n'
        page.write_text(original, encoding="utf-8")

        _add_aliases(page, [])
        assert page.read_text(encoding="utf-8") == original

    def test_nonexistent_file_does_nothing(self, tmp_path):
        path = tmp_path / "nonexistent.md"
        # Should not raise
        _add_aliases(path, ["alias"])

    def test_file_without_frontmatter_does_nothing(self, tmp_path):
        page = tmp_path / "no-fm.md"
        original = "# No Frontmatter\n\nJust content.\n"
        page.write_text(original, encoding="utf-8")

        _add_aliases(page, ["alias"])
        assert page.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# _execute_merge
# ---------------------------------------------------------------------------


class TestExecuteMerge:
    def test_renames_duplicate_to_canonical(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        subdir = wiki_dir / "articles"
        subdir.mkdir(parents=True)

        # File must match _safe_slug("Risk Management") = "risk management"
        dup_content = '---\ntitle: "Risk Management"\n---\n# Risk Management\n\nContent.\n'
        (subdir / "risk management.md").write_text(dup_content, encoding="utf-8")

        group = {"canonical": "Credit Risk", "duplicates": ["Risk Management"]}
        result = _execute_merge(wiki_dir, subdir, group)

        # _safe_slug("Credit Risk") = "credit risk"
        canonical_path = subdir / "credit risk.md"
        assert canonical_path.exists()
        assert result["canonical"] == "Credit Risk"

    def test_removes_duplicate_file(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        subdir = wiki_dir / "articles"
        subdir.mkdir(parents=True)

        # _safe_slug("Credit Risks") = "credit risks"
        (subdir / "credit risks.md").write_text(
            '---\ntitle: "Credit Risks"\n---\n# Credit Risks\n\nBody.\n', encoding="utf-8"
        )
        group = {"canonical": "Credit Risk", "duplicates": ["Credit Risks"]}
        _execute_merge(wiki_dir, subdir, group)

        assert not (subdir / "credit risks.md").exists()

    def test_replaces_wikilinks_after_merge(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        subdir = wiki_dir / "articles"
        subdir.mkdir(parents=True)

        (subdir / "credit risks.md").write_text(
            '---\ntitle: "Credit Risks"\n---\n# Credit Risks\n\nBody.\n', encoding="utf-8"
        )
        ref_page = wiki_dir / "reference.md"
        # Link uses the slug form of the dup
        ref_page.write_text("See [[credit risks]] for info.", encoding="utf-8")

        group = {"canonical": "Credit Risk", "duplicates": ["Credit Risks"]}
        _execute_merge(wiki_dir, subdir, group)

        ref_content = ref_page.read_text(encoding="utf-8")
        assert "[[credit risk]]" in ref_content

    def test_skips_when_canonical_already_exists(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        subdir = wiki_dir / "articles"
        subdir.mkdir(parents=True)

        canon_content = '---\ntitle: "Credit Risk"\n---\n# Credit Risk\n\nOriginal.\n'
        (subdir / "credit risk.md").write_text(canon_content, encoding="utf-8")
        (subdir / "credit risks.md").write_text(
            '---\ntitle: "Credit Risks"\n---\n# Dup\n', encoding="utf-8"
        )

        group = {"canonical": "Credit Risk", "duplicates": ["Credit Risks"]}
        _execute_merge(wiki_dir, subdir, group)

        # Canonical should still exist and dup should be removed
        assert (subdir / "credit risk.md").exists()
        assert not (subdir / "credit risks.md").exists()
        # Content body is preserved
        final_content = (subdir / "credit risk.md").read_text(encoding="utf-8")
        assert "Original." in final_content


# ---------------------------------------------------------------------------
# _identify_groups + run_consolidate (async, with mock LLM)
# ---------------------------------------------------------------------------


class TestRunConsolidate:
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
    async def test_run_consolidate_no_pages(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.consolidate import run_consolidate

        cfg = self._make_cfg(tmp_path)
        # No articles dir at all
        mock_llm = MagicMock()
        mock_logger = MagicMock()

        await run_consolidate(cfg, mock_llm, mock_logger)
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_consolidate_with_duplicate_group(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.llm.base import LLMResponse
        from src.stages.consolidate import run_consolidate
        import json

        cfg = self._make_cfg(tmp_path)
        articles = cfg.wiki_dir / "articles"
        articles.mkdir()
        (articles / "credit risk.md").write_text(
            '---\ntitle: "Credit Risk"\n---\n# Credit Risk\n\nContent.\n'
        )
        (articles / "credit risks.md").write_text(
            '---\ntitle: "Credit Risks"\n---\n# Credit Risks\n\nContent.\n'
        )

        groups_resp = json.dumps([{
            "canonical": "Credit Risk",
            "duplicates": ["Credit Risks"]
        }])
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=LLMResponse(
            text=groups_resp, tokens_in=10, tokens_out=5,
            cached_tokens=None, model_id="test", attempts=1,
        ))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        await run_consolidate(cfg, mock_llm, mock_logger)

        # Duplicate should be gone
        assert not (articles / "credit risks.md").exists()
        assert (articles / "credit risk.md").exists()

    @pytest.mark.asyncio
    async def test_run_consolidate_handles_llm_error(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.consolidate import run_consolidate

        cfg = self._make_cfg(tmp_path)
        articles = cfg.wiki_dir / "articles"
        articles.mkdir()
        for i in range(3):
            (articles / f"page-{i}.md").write_text(f"# Page {i}\n\nContent.\n")

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(side_effect=RuntimeError("LLM failed"))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        # Should not raise — handles gracefully
        await run_consolidate(cfg, mock_llm, mock_logger)
