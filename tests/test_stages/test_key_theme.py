"""Unit tests for src/stages/key_theme.py — pure helper functions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.models.config import KeyThemeConfig
from src.stages.key_theme import (
    _collect_terms_from_frontmatter,
    _collect_terms_from_sections,
    _safe_slug,
    _section_pattern,
)


# ---------------------------------------------------------------------------
# _safe_slug
# ---------------------------------------------------------------------------


class TestSafeSlug:
    def test_lowercases(self):
        result = _safe_slug("Machine Learning")
        assert result == "machine learning"

    def test_replaces_invalid_chars(self):
        result = _safe_slug("AI/ML")
        assert "/" not in result

    def test_empty_returns_page(self):
        assert _safe_slug("") == "page"

    def test_keeps_spaces(self):
        result = _safe_slug("Natural Language Processing")
        assert result == "natural language processing"


# ---------------------------------------------------------------------------
# _section_pattern
# ---------------------------------------------------------------------------


class TestSectionPattern:
    def test_compiles_regex(self):
        pattern = _section_pattern("## Topics")
        assert hasattr(pattern, "search")

    def test_matches_exact_section(self):
        pattern = _section_pattern("## Topics")
        text = "## Topics\n\n[[link-a]] [[link-b]]\n\n## Another Section\n"
        m = pattern.search(text)
        assert m is not None
        assert "link-a" in m.group(1)

    def test_case_insensitive(self):
        pattern = _section_pattern("## topics")
        text = "## TOPICS\n\n[[link-a]]\n"
        m = pattern.search(text)
        assert m is not None

    def test_stops_at_next_heading(self):
        pattern = _section_pattern("## Topics")
        text = "## Topics\n\n[[good-link]]\n\n## Other\n\n[[bad-link]]\n"
        m = pattern.search(text)
        assert m is not None
        section_body = m.group(1)
        assert "good-link" in section_body
        assert "bad-link" not in section_body

    def test_non_matching_section(self):
        pattern = _section_pattern("## Topics")
        text = "## Different Section\n\nContent.\n"
        m = pattern.search(text)
        assert m is None


# ---------------------------------------------------------------------------
# _collect_terms_from_sections
# ---------------------------------------------------------------------------


class TestCollectTermsFromSections:
    def _make_theme_cfg(self, tmp_path: Path) -> KeyThemeConfig:
        n = tmp_path / "norm.md"
        n.write_text("prompt")
        c = tmp_path / "create.md"
        c.write_text("prompt")
        return KeyThemeConfig(
            name="Topics",
            wiki_subdir="topics",
            term_source="section_wikilinks",
            section_header="## Topics",
            prompt_normalize=n,
            prompt_create_page=c,
        )

    def test_collects_wikilinks_from_section(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "article.md").write_text(
            "# Article\n\n## Topics\n\n[[Machine Learning]] [[NLP]]\n\n## Other\n\ncontent\n",
            encoding="utf-8",
        )
        theme_cfg = self._make_theme_cfg(tmp_path)
        terms = _collect_terms_from_sections(wiki_dir, theme_cfg)
        assert "Machine Learning" in terms
        assert "NLP" in terms

    def test_records_page_id_for_each_term(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "my-article.md").write_text(
            "# Article\n\n## Topics\n\n[[Deep Learning]]\n",
            encoding="utf-8",
        )
        theme_cfg = self._make_theme_cfg(tmp_path)
        terms = _collect_terms_from_sections(wiki_dir, theme_cfg)
        assert "my-article" in terms.get("Deep Learning", [])

    def test_skips_pages_without_section(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text(
            "# Page\n\n## Different Section\n\n[[link]]\n",
            encoding="utf-8",
        )
        theme_cfg = self._make_theme_cfg(tmp_path)
        terms = _collect_terms_from_sections(wiki_dir, theme_cfg)
        assert len(terms) == 0

    def test_skips_theme_subdir_pages(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        theme_subdir = wiki_dir / "topics"
        theme_subdir.mkdir(parents=True)
        (theme_subdir / "topic-page.md").write_text(
            "# Topic\n\n## Topics\n\n[[Self Link]]\n",
            encoding="utf-8",
        )
        theme_cfg = self._make_theme_cfg(tmp_path)
        terms = _collect_terms_from_sections(wiki_dir, theme_cfg)
        assert "Self Link" not in terms

    def test_term_appears_in_multiple_pages(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        for i in range(3):
            (wiki_dir / f"article-{i}.md").write_text(
                f"# Article {i}\n\n## Topics\n\n[[Common Topic]]\n",
                encoding="utf-8",
            )
        theme_cfg = self._make_theme_cfg(tmp_path)
        terms = _collect_terms_from_sections(wiki_dir, theme_cfg)
        assert len(terms["Common Topic"]) == 3


# ---------------------------------------------------------------------------
# _collect_terms_from_frontmatter
# ---------------------------------------------------------------------------


class TestCollectTermsFromFrontmatter:
    def _make_theme_cfg(self, tmp_path: Path) -> KeyThemeConfig:
        n = tmp_path / "norm.md"
        n.write_text("prompt")
        c = tmp_path / "create.md"
        c.write_text("prompt")
        return KeyThemeConfig(
            name="Tags",
            wiki_subdir="tags",
            term_source="metadata_field",
            metadata_field="tags",
            prompt_normalize=n,
            prompt_create_page=c,
        )

    def test_collects_scalar_value(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text(
            "---\ntags: machine-learning\n---\n# Page\n",
            encoding="utf-8",
        )
        theme_cfg = self._make_theme_cfg(tmp_path)
        terms = _collect_terms_from_frontmatter(wiki_dir, theme_cfg)
        assert "machine-learning" in terms

    def test_skips_pages_without_field(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text(
            "---\ntitle: No Tags\n---\n# Page\n",
            encoding="utf-8",
        )
        theme_cfg = self._make_theme_cfg(tmp_path)
        terms = _collect_terms_from_frontmatter(wiki_dir, theme_cfg)
        assert len(terms) == 0


# ---------------------------------------------------------------------------
# normalize_terms (async, with mock LLM)
# ---------------------------------------------------------------------------


class TestNormalizeTerms:
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

    def _make_theme_cfg(self, tmp_path: Path) -> KeyThemeConfig:
        n = tmp_path / "norm.md"; n.write_text("Normalize: {{ language }}")
        c = tmp_path / "create.md"; c.write_text("Create: {{ term }}")
        return KeyThemeConfig(
            name="Topics", wiki_subdir="topics",
            term_source="section_wikilinks",
            section_header="## Topics",
            prompt_normalize=n,
            prompt_create_page=c,
        )

    @pytest.mark.asyncio
    async def test_normalize_terms_empty(self, tmp_path):
        from unittest.mock import MagicMock
        from src.stages.key_theme import normalize_terms

        cfg = self._make_cfg(tmp_path)
        theme_cfg = self._make_theme_cfg(tmp_path)
        mock_llm = MagicMock()
        mock_logger = MagicMock()

        result = await normalize_terms({}, theme_cfg, cfg, mock_llm, mock_logger)
        assert result == {}
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_normalize_terms_returns_mapping(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.llm.base import LLMResponse
        from src.stages.key_theme import normalize_terms
        import json

        cfg = self._make_cfg(tmp_path)
        theme_cfg = self._make_theme_cfg(tmp_path)

        terms = {"machine learning": ["page-a"], "ML": ["page-b"]}
        mapping_json = json.dumps({"machine learning": "Machine Learning", "ML": "Machine Learning"})

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=LLMResponse(
            text=mapping_json, tokens_in=5, tokens_out=5,
            cached_tokens=None, model_id="test", attempts=1,
        ))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await normalize_terms(terms, theme_cfg, cfg, mock_llm, mock_logger)
        assert "machine learning" in result
        assert result["ML"] == "Machine Learning"

    @pytest.mark.asyncio
    async def test_normalize_terms_falls_back_on_llm_error(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.key_theme import normalize_terms

        cfg = self._make_cfg(tmp_path)
        theme_cfg = self._make_theme_cfg(tmp_path)

        terms = {"nlp": ["page-a"]}
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        result = await normalize_terms(terms, theme_cfg, cfg, mock_llm, mock_logger)
        # Falls back to identity mapping
        assert result.get("nlp") == "nlp"


# ---------------------------------------------------------------------------
# collect_terms dispatch
# ---------------------------------------------------------------------------


class TestCollectTerms:
    def _make_section_cfg(self, tmp_path: Path) -> KeyThemeConfig:
        n = tmp_path / "n.md"; n.write_text("prompt")
        c = tmp_path / "c.md"; c.write_text("prompt")
        return KeyThemeConfig(
            name="T", wiki_subdir="t", term_source="section_wikilinks",
            section_header="## Topics", prompt_normalize=n, prompt_create_page=c,
        )

    def _make_fm_cfg(self, tmp_path: Path) -> KeyThemeConfig:
        n = tmp_path / "n2.md"; n.write_text("prompt")
        c = tmp_path / "c2.md"; c.write_text("prompt")
        return KeyThemeConfig(
            name="T2", wiki_subdir="t2", term_source="metadata_field",
            metadata_field="tags", prompt_normalize=n, prompt_create_page=c,
        )

    def test_dispatch_sections(self, tmp_path):
        from src.stages.key_theme import collect_terms
        wiki_dir = tmp_path / "wiki"; wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text("# P\n\n## Topics\n\n[[Tag A]]\n")
        cfg = self._make_section_cfg(tmp_path)
        result = collect_terms(wiki_dir, cfg)
        assert "Tag A" in result

    def test_dispatch_frontmatter(self, tmp_path):
        from src.stages.key_theme import collect_terms
        wiki_dir = tmp_path / "wiki"; wiki_dir.mkdir()
        (wiki_dir / "page.md").write_text("---\ntags: python\n---\n# P\n")
        cfg = self._make_fm_cfg(tmp_path)
        result = collect_terms(wiki_dir, cfg)
        assert "python" in result


# ---------------------------------------------------------------------------
# generate_key_theme_pages (async)
# ---------------------------------------------------------------------------


class TestGenerateKeyThemePages:
    def _make_cfg(self, tmp_path):
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

    def _make_theme_cfg(self, tmp_path):
        n = tmp_path / "n.md"; n.write_text("Normalize: {{ language }}")
        c = tmp_path / "c.md"; c.write_text("Create page for {{ term }}")
        return KeyThemeConfig(
            name="Topics", wiki_subdir="topics", term_source="section_wikilinks",
            section_header="## Topics", prompt_normalize=n, prompt_create_page=c,
        )

    @pytest.mark.asyncio
    async def test_generates_pages_for_terms(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.llm.base import LLMResponse
        from src.stages.key_theme import generate_key_theme_pages

        cfg = self._make_cfg(tmp_path)
        theme_cfg = self._make_theme_cfg(tmp_path)

        terms = {"Machine Learning": ["page-a"]}
        mapping = {"Machine Learning": "Machine Learning"}

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=LLMResponse(
            text="# Machine Learning\n\nContent.", tokens_in=5, tokens_out=10,
            cached_tokens=None, model_id="test", attempts=1,
        ))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        await generate_key_theme_pages(terms, mapping, theme_cfg, cfg, mock_llm, mock_logger)

        dest = cfg.wiki_dir / "topics" / "machine learning.md"
        assert dest.exists()
        assert "Machine Learning" in dest.read_text()

    @pytest.mark.asyncio
    async def test_skips_existing_pages(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.key_theme import generate_key_theme_pages

        cfg = self._make_cfg(tmp_path)
        theme_cfg = self._make_theme_cfg(tmp_path)

        topic_dir = cfg.wiki_dir / "topics"
        topic_dir.mkdir(parents=True)
        existing = topic_dir / "nlp.md"
        existing.write_text("# NLP\n\nExisting content.\n")

        terms = {"NLP": ["page-a"]}
        mapping = {"NLP": "NLP"}

        mock_llm = MagicMock()
        mock_logger = MagicMock()

        await generate_key_theme_pages(terms, mapping, theme_cfg, cfg, mock_llm, mock_logger)
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock
        from src.stages.key_theme import generate_key_theme_pages

        cfg = self._make_cfg(tmp_path)
        theme_cfg = self._make_theme_cfg(tmp_path)

        terms = {"Deep Learning": ["page-a"]}
        mapping = {"Deep Learning": "Deep Learning"}

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(side_effect=RuntimeError("LLM failed"))
        mock_logger = MagicMock()
        mock_logger.start_call.return_value = 0.0

        # Should not raise
        await generate_key_theme_pages(terms, mapping, theme_cfg, cfg, mock_llm, mock_logger)


# ---------------------------------------------------------------------------
# run_key_themes (async)
# ---------------------------------------------------------------------------


class TestRunKeyThemes:
    def _make_cfg_with_key_themes(self, tmp_path):
        from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig
        wiki = tmp_path / "wiki"; wiki.mkdir()
        logs = tmp_path / "logs"; logs.mkdir()
        prompts = tmp_path / "prompts"; prompts.mkdir()
        for fn in ["gen.md", "eval.md", "editor.md", "lint.md", "consol.md", "chat.md"]:
            (prompts / fn).write_text("prompt text")
        norm = prompts / "norm.md"; norm.write_text("Normalize: {{ language }}")
        create = prompts / "create.md"; create.write_text("Create: {{ term }}")
        theme = KeyThemeConfig(
            name="Topics", wiki_subdir="topics", term_source="section_wikilinks",
            section_header="## Topics", prompt_normalize=norm, prompt_create_page=create,
        )
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
            key_themes=[theme],
        )

    @pytest.mark.asyncio
    async def test_run_key_themes_no_config(self, tmp_path):
        from unittest.mock import MagicMock
        from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig
        from src.stages.key_theme import run_key_themes

        wiki = tmp_path / "wiki"; wiki.mkdir()
        logs = tmp_path / "logs"; logs.mkdir()
        prompts = tmp_path / "prompts"; prompts.mkdir()
        for fn in ["gen.md", "eval.md", "editor.md", "lint.md", "consol.md", "chat.md"]:
            (prompts / fn).write_text("prompt text")
        cfg = WikiConfig(
            wiki_name="Test", wiki_dir=wiki, log_dir=logs,
            llm=LLMConfig(backend="openrouter", model_id="test"),
            objects=[ObjectTypeConfig(
                name="Article", slug="article", wiki_subdir="articles",
                prompt_generate=prompts / "gen.md", prompt_evaluate=prompts / "eval.md",
            )],
            prompt_editor=prompts / "editor.md",
            prompt_lint=prompts / "lint.md",
            prompt_consolidate=prompts / "consol.md",
            prompt_chat=prompts / "chat.md",
        )

        mock_llm = MagicMock()
        mock_logger = MagicMock()
        await run_key_themes(cfg, mock_llm, mock_logger)
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_key_themes_with_config(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.stages.key_theme import run_key_themes

        cfg = self._make_cfg_with_key_themes(tmp_path)
        mock_llm = MagicMock()
        mock_logger = MagicMock()

        with patch("src.stages.key_theme.collect_terms", return_value={}), \
             patch("src.stages.key_theme.normalize_terms", new=AsyncMock(return_value={})), \
             patch("src.stages.key_theme.generate_key_theme_pages", new=AsyncMock()):
            await run_key_themes(cfg, mock_llm, mock_logger)
