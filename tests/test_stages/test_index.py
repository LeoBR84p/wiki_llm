"""Unit tests for src/stages/index.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stages._utils import collect_wiki_pages
from src.stages.index import run_index


class TestCollectWikiPages:
    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        pages = collect_wiki_pages(tmp_path / "nonexistent")
        assert pages == []

    def test_returns_sorted_md_files(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "b-page.md").write_text("content b")
        (subdir / "a-page.md").write_text("content a")
        (subdir / "c-page.md").write_text("content c")

        pages = collect_wiki_pages(subdir)
        names = [p.name for p in pages]
        assert names == sorted(names)
        assert len(names) == 3

    def test_excludes_system_pages(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "index.md").write_text("index content")
        (subdir / "log.md").write_text("log content")
        (subdir / "lint_report.md").write_text("lint content")
        (subdir / "my-article.md").write_text("article content")

        pages = collect_wiki_pages(subdir)
        names = [p.name for p in pages]
        assert "index.md" not in names
        assert "log.md" not in names
        assert "lint_report.md" not in names
        assert "my-article.md" in names

    def test_only_returns_md_files(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "article.md").write_text("md content")
        (subdir / "image.png").write_bytes(b"png")
        (subdir / "data.txt").write_text("txt")

        pages = collect_wiki_pages(subdir)
        assert all(p.suffix == ".md" for p in pages)
        assert len(pages) == 1

    def test_empty_directory(self, tmp_path):
        subdir = tmp_path / "empty"
        subdir.mkdir()
        assert collect_wiki_pages(subdir) == []

    def test_custom_skip_set(self, tmp_path):
        subdir = tmp_path / "articles"
        subdir.mkdir()
        (subdir / "skip-me.md").write_text("skip")
        (subdir / "keep-me.md").write_text("keep")

        pages = collect_wiki_pages(subdir, skip=frozenset({"skip-me.md"}))
        names = [p.name for p in pages]
        assert "skip-me.md" not in names
        assert "keep-me.md" in names


class TestRunIndex:
    @pytest.mark.asyncio
    async def test_creates_index_md(self, minimal_config):
        cfg = minimal_config
        articles_dir = cfg.wiki_dir / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "my-article.md").write_text("# My Article\n", encoding="utf-8")

        await run_index(cfg)

        index_path = cfg.wiki_dir / "index.md"
        assert index_path.exists()

    @pytest.mark.asyncio
    async def test_index_contains_frontmatter(self, minimal_config):
        cfg = minimal_config
        articles_dir = cfg.wiki_dir / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "article-one.md").write_text("# Article One\n")

        await run_index(cfg)
        content = (cfg.wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "type: index" in content

    @pytest.mark.asyncio
    async def test_index_lists_object_type_pages(self, minimal_config):
        cfg = minimal_config
        articles_dir = cfg.wiki_dir / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "article-alpha.md").write_text("# Alpha\n")
        (articles_dir / "article-beta.md").write_text("# Beta\n")

        await run_index(cfg)
        content = (cfg.wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "[[article-alpha]]" in content
        assert "[[article-beta]]" in content

    @pytest.mark.asyncio
    async def test_index_skips_empty_subdirs(self, minimal_config):
        cfg = minimal_config
        # Do NOT create the articles subdir or any pages
        await run_index(cfg)
        content = (cfg.wiki_dir / "index.md").read_text(encoding="utf-8")
        # Section for "Article" should NOT appear since there are no pages
        assert "## Article\n" not in content

    @pytest.mark.asyncio
    async def test_index_with_key_themes(self, config_with_themes_and_groups):
        cfg = config_with_themes_and_groups
        topics_dir = cfg.wiki_dir / "topics"
        topics_dir.mkdir(parents=True)
        (topics_dir / "machine-learning.md").write_text("# ML\n")

        await run_index(cfg)
        content = (cfg.wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "[[machine-learning]]" in content

    @pytest.mark.asyncio
    async def test_index_with_groups(self, config_with_themes_and_groups):
        cfg = config_with_themes_and_groups
        teams_dir = cfg.wiki_dir / "teams"
        teams_dir.mkdir(parents=True)
        (teams_dir / "engineering.md").write_text("# Engineering\n")

        await run_index(cfg)
        content = (cfg.wiki_dir / "index.md").read_text(encoding="utf-8")
        assert "[[engineering]]" in content

    @pytest.mark.asyncio
    async def test_index_is_idempotent(self, minimal_config):
        cfg = minimal_config
        articles_dir = cfg.wiki_dir / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "my-page.md").write_text("# My Page\n")

        await run_index(cfg)
        content_first = (cfg.wiki_dir / "index.md").read_text(encoding="utf-8")
        await run_index(cfg)
        content_second = (cfg.wiki_dir / "index.md").read_text(encoding="utf-8")
        assert content_first == content_second
