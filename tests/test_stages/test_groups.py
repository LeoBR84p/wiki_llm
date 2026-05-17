"""Unit tests for src/stages/groups.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.config import GroupConfig, LLMConfig, ObjectTypeConfig, WikiConfig
from src.models.document import Document, DocumentMetadata
from src.stages.groups import (
    _group_value,
    _page_content,
    _safe_slug,
    run_groups,
)


# ---------------------------------------------------------------------------
# _safe_slug
# ---------------------------------------------------------------------------


class TestSafeSlug:
    def test_lowercases(self):
        assert _safe_slug("Engineering Team") == "engineering team"

    def test_replaces_invalid_chars(self):
        result = _safe_slug("Finance/HR")
        assert "/" not in result
        assert result == "finance-hr"

    def test_empty_string_returns_page(self):
        assert _safe_slug("") == "page"

    def test_colon_removed(self):
        result = _safe_slug("A: B")
        assert ":" not in result

    def test_keeps_spaces(self):
        result = _safe_slug("Team Alpha 2024")
        assert result == "team alpha 2024"


# ---------------------------------------------------------------------------
# _group_value
# ---------------------------------------------------------------------------


class TestGroupValue:
    def _make_doc(self, extra: dict) -> Document:
        return Document(
            metadata=DocumentMetadata(
                id="abc",
                title="Test",
                object_type="article",
                extra=extra,
            ),
            content="content",
        )

    def test_reads_from_extra(self):
        doc = self._make_doc({"team": "Engineering"})
        val = _group_value(doc, "team")
        assert val == "Engineering"

    def test_returns_none_for_missing_field(self):
        doc = self._make_doc({})
        val = _group_value(doc, "team")
        assert val is None

    def test_returns_none_for_empty_value(self):
        doc = self._make_doc({"team": ""})
        val = _group_value(doc, "team")
        assert val is None

    def test_strips_whitespace(self):
        doc = self._make_doc({"team": "  Finance  "})
        val = _group_value(doc, "team")
        assert val == "Finance"

    def test_reads_from_metadata_attribute(self):
        # object_type is an attribute of metadata (not in extra)
        doc = self._make_doc({})
        val = _group_value(doc, "object_type")
        assert val == "article"


# ---------------------------------------------------------------------------
# _page_content
# ---------------------------------------------------------------------------


class TestPageContent:
    def _make_docs(self, n: int) -> list[Document]:
        return [
            Document(
                metadata=DocumentMetadata(
                    id=f"id-{i:04d}",
                    title=f"Document {i}",
                    object_type="article",
                    status="active",
                ),
                content=f"Content {i}",
            )
            for i in range(n)
        ]

    def _make_grp_cfg(self) -> GroupConfig:
        return GroupConfig(name="Team", wiki_subdir="teams", metadata_field="team")

    def test_includes_frontmatter(self):
        docs = self._make_docs(1)
        grp_cfg = self._make_grp_cfg()
        content = _page_content("Engineering", "team", docs, grp_cfg)
        assert "type: group" in content
        assert 'value: "Engineering"' in content

    def test_includes_document_ids(self):
        docs = self._make_docs(2)
        grp_cfg = self._make_grp_cfg()
        content = _page_content("Engineering", "team", docs, grp_cfg)
        assert "id-0000" in content
        assert "id-0001" in content

    def test_heading_contains_group_name(self):
        docs = self._make_docs(1)
        grp_cfg = self._make_grp_cfg()
        content = _page_content("Finance", "team", docs, grp_cfg)
        assert "# Team: Finance" in content

    def test_total_count_in_frontmatter(self):
        docs = self._make_docs(3)
        grp_cfg = self._make_grp_cfg()
        content = _page_content("Engineering", "team", docs, grp_cfg)
        assert "total: 3" in content

    def test_table_headers_present(self):
        docs = self._make_docs(1)
        grp_cfg = self._make_grp_cfg()
        content = _page_content("Team", "team", docs, grp_cfg)
        assert "| id | title | status |" in content

    def test_documents_sorted_by_id(self):
        docs = self._make_docs(3)
        grp_cfg = self._make_grp_cfg()
        content = _page_content("Team", "team", docs, grp_cfg)
        # ids are id-0000, id-0001, id-0002 — sorted order
        pos0 = content.index("id-0000")
        pos1 = content.index("id-0001")
        pos2 = content.index("id-0002")
        assert pos0 < pos1 < pos2


# ---------------------------------------------------------------------------
# run_groups
# ---------------------------------------------------------------------------


class TestRunGroups:
    @pytest.mark.asyncio
    async def test_does_nothing_with_no_groups(self, minimal_config):
        """When no groups are configured, run_groups is a no-op."""
        docs = [
            Document(
                metadata=DocumentMetadata(id="1", title="T", object_type="article"),
                content="content",
            )
        ]
        await run_groups(minimal_config, docs)
        # No group pages should be created
        assert not (minimal_config.wiki_dir / "teams").exists()

    @pytest.mark.asyncio
    async def test_creates_group_pages(self, config_with_themes_and_groups):
        cfg = config_with_themes_and_groups
        docs = [
            Document(
                metadata=DocumentMetadata(
                    id="doc-1",
                    title="Doc One",
                    object_type="article",
                    extra={"team": "Engineering"},
                ),
                content="content",
            ),
            Document(
                metadata=DocumentMetadata(
                    id="doc-2",
                    title="Doc Two",
                    object_type="article",
                    extra={"team": "Finance"},
                ),
                content="content",
            ),
        ]
        await run_groups(cfg, docs)

        teams_dir = cfg.wiki_dir / "teams"
        assert teams_dir.exists()
        pages = list(teams_dir.glob("*.md"))
        assert len(pages) == 2

    @pytest.mark.asyncio
    async def test_skips_existing_group_pages(self, config_with_themes_and_groups):
        cfg = config_with_themes_and_groups
        teams_dir = cfg.wiki_dir / "teams"
        teams_dir.mkdir(parents=True)
        existing = teams_dir / "Engineering.md"
        existing.write_text("# Existing page\n", encoding="utf-8")

        docs = [
            Document(
                metadata=DocumentMetadata(
                    id="doc-1",
                    title="Doc One",
                    object_type="article",
                    extra={"team": "Engineering"},
                ),
                content="content",
            )
        ]
        await run_groups(cfg, docs)

        # Existing page should not be overwritten
        assert existing.read_text(encoding="utf-8") == "# Existing page\n"

    @pytest.mark.asyncio
    async def test_docs_without_group_field_are_ignored(self, config_with_themes_and_groups):
        cfg = config_with_themes_and_groups
        docs = [
            Document(
                metadata=DocumentMetadata(
                    id="doc-1",
                    title="No Team Doc",
                    object_type="article",
                    extra={},  # no "team" field
                ),
                content="content",
            )
        ]
        await run_groups(cfg, docs)
        teams_dir = cfg.wiki_dir / "teams"
        pages = list(teams_dir.glob("*.md")) if teams_dir.exists() else []
        assert len(pages) == 0
