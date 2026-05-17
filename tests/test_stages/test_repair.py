"""Unit tests for src/stages/repair.py."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.evaluation import RepairState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path):
    from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig

    wiki = tmp_path / "wiki"
    wiki.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for fn in ["gen.md", "eval.md", "editor.md", "lint.md", "consol.md", "chat.md"]:
        (prompts / fn).write_text("Handle repair for: {{ target }}")
    return WikiConfig(
        wiki_name="Test",
        wiki_dir=wiki,
        log_dir=logs,
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


def _make_llm_response(text: str = "Stub content."):
    from src.llm.base import LLMResponse

    return LLMResponse(
        text=text,
        tokens_in=5,
        tokens_out=10,
        cached_tokens=None,
        model_id="test",
        attempts=1,
    )


def _make_logger():
    mock = MagicMock()
    mock.start_call.return_value = 0.0
    return mock


# ---------------------------------------------------------------------------
# _find_page
# ---------------------------------------------------------------------------


class TestFindPage:
    def test_finds_existing_file(self, tmp_path):
        from src.stages.repair import _find_page

        (tmp_path / "my-page.md").write_text("# Page")
        result = _find_page(tmp_path, "my-page")
        assert result is not None
        assert result.name == "my-page.md"

    def test_returns_none_when_missing(self, tmp_path):
        from src.stages.repair import _find_page

        result = _find_page(tmp_path, "nonexistent")
        assert result is None

    def test_finds_in_subdirectory(self, tmp_path):
        from src.stages.repair import _find_page

        sub = tmp_path / "articles"
        sub.mkdir()
        (sub / "deep-page.md").write_text("# Deep Page")
        result = _find_page(tmp_path, "deep-page")
        assert result is not None
        assert result.name == "deep-page.md"


# ---------------------------------------------------------------------------
# _add_link_to_page
# ---------------------------------------------------------------------------


class TestAddLinkToPage:
    @pytest.mark.asyncio
    async def test_adds_link_to_existing_page(self, tmp_path):
        from src.stages.repair import _add_link_to_page

        page = tmp_path / "target.md"
        page.write_text("# Target Page\n\nContent.", encoding="utf-8")
        file_locks: dict = {}

        added = await _add_link_to_page(tmp_path, "target", "source-page", file_locks)
        assert added is True
        assert "[[source-page]]" in page.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_skips_when_link_already_present(self, tmp_path):
        from src.stages.repair import _add_link_to_page

        page = tmp_path / "target.md"
        page.write_text("# Target\n\n[[source-page]]\n", encoding="utf-8")
        file_locks: dict = {}

        added = await _add_link_to_page(tmp_path, "target", "source-page", file_locks)
        assert added is False

    @pytest.mark.asyncio
    async def test_returns_false_when_page_not_found(self, tmp_path):
        from src.stages.repair import _add_link_to_page

        file_locks: dict = {}
        added = await _add_link_to_page(tmp_path, "nonexistent", "source", file_locks)
        assert added is False

    @pytest.mark.asyncio
    async def test_creates_lock_for_new_page(self, tmp_path):
        from src.stages.repair import _add_link_to_page

        page = tmp_path / "new-target.md"
        page.write_text("# New Target\n\nContent.", encoding="utf-8")
        file_locks: dict = {}

        await _add_link_to_page(tmp_path, "new-target", "linker", file_locks)
        # Lock was created for the page
        assert len(file_locks) == 1


# ---------------------------------------------------------------------------
# _build_item_states
# ---------------------------------------------------------------------------


class TestBuildItemStates:
    def _make_state(self, orphans=None, broken_links=None, wiki_dir="/tmp/wiki"):
        return {
            "wiki_dir": wiki_dir,
            "orphans": orphans or [],
            "broken_links": broken_links or [],
            "repaired": [],
            "errors": [],
        }

    def test_empty_state_returns_empty_list(self):
        from src.stages.repair import _build_item_states

        result = _build_item_states(self._make_state())
        assert result == []

    def test_orphans_become_items(self):
        from src.stages.repair import _build_item_states

        state = self._make_state(orphans=["page-a", "page-b"])
        items = _build_item_states(state)
        orphan_items = [i for i in items if i["repair_type"] == "orphan"]
        assert len(orphan_items) == 2
        assert {i["target"] for i in orphan_items} == {"page-a", "page-b"}

    def test_broken_links_grouped_by_target(self):
        from src.stages.repair import _build_item_states

        state = self._make_state(broken_links=[
            {"source": "page-a", "target": "missing"},
            {"source": "page-b", "target": "missing"},
            {"source": "page-c", "target": "other-missing"},
        ])
        items = _build_item_states(state)
        bl_items = [i for i in items if i["repair_type"] == "broken_link"]
        assert len(bl_items) == 2  # 2 unique targets
        missing_item = next(i for i in bl_items if i["target"] == "missing")
        assert set(missing_item["sources"]) == {"page-a", "page-b"}

    def test_mixed_orphans_and_broken_links(self):
        from src.stages.repair import _build_item_states

        state = self._make_state(
            orphans=["lone-page"],
            broken_links=[{"source": "page-x", "target": "ghost"}],
        )
        items = _build_item_states(state)
        assert len(items) == 2
        types = {i["repair_type"] for i in items}
        assert types == {"orphan", "broken_link"}

    def test_item_wiki_dir_matches_state(self):
        from src.stages.repair import _build_item_states

        state = self._make_state(orphans=["page"], wiki_dir="/some/wiki")
        items = _build_item_states(state)
        assert items[0]["wiki_dir"] == "/some/wiki"


# ---------------------------------------------------------------------------
# _repair_broken_link
# ---------------------------------------------------------------------------


class TestRepairBrokenLink:
    @pytest.mark.asyncio
    async def test_creates_stub_page(self, tmp_path):
        from src.stages.repair import _repair_broken_link

        cfg = _make_cfg(tmp_path)
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=_make_llm_response("This is a stub page."))
        mock_logger = _make_logger()
        file_locks: dict = {}
        creating_pages: set = set()
        creating_lock = asyncio.Lock()

        repaired, errors = await _repair_broken_link(
            cfg.wiki_dir, "Missing Concept", ["page-a", "page-b"],
            cfg, mock_llm, mock_logger,
            creating_pages, creating_lock, file_locks,
        )

        assert repaired == ["Missing Concept"]
        assert errors == []
        stub = cfg.wiki_dir / "missing concept.md"
        assert stub.exists()
        content = stub.read_text(encoding="utf-8")
        assert "Missing Concept" in content

    @pytest.mark.asyncio
    async def test_skips_if_page_already_exists(self, tmp_path):
        from src.stages.repair import _repair_broken_link

        cfg = _make_cfg(tmp_path)
        existing = cfg.wiki_dir / "existing page.md"
        existing.write_text("# Existing Page\n\nAlready here.", encoding="utf-8")

        mock_llm = MagicMock()
        mock_logger = _make_logger()
        file_locks: dict = {}
        creating_pages: set = set()
        creating_lock = asyncio.Lock()

        repaired, errors = await _repair_broken_link(
            cfg.wiki_dir, "Existing Page", ["page-a"],
            cfg, mock_llm, mock_logger,
            creating_pages, creating_lock, file_locks,
        )

        assert repaired == []
        assert errors == []
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_failure(self, tmp_path):
        from src.stages.repair import _repair_broken_link

        cfg = _make_cfg(tmp_path)
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_logger = _make_logger()
        file_locks: dict = {}
        creating_pages: set = set()
        creating_lock = asyncio.Lock()

        repaired, errors = await _repair_broken_link(
            cfg.wiki_dir, "Ghost Page", ["page-x"],
            cfg, mock_llm, mock_logger,
            creating_pages, creating_lock, file_locks,
        )

        assert repaired == []
        assert len(errors) == 1
        assert "Ghost Page" in errors[0]

    @pytest.mark.asyncio
    async def test_skips_if_slug_in_creating_pages(self, tmp_path):
        from src.stages.repair import _repair_broken_link
        from src.stages._utils import _safe_slug

        cfg = _make_cfg(tmp_path)
        mock_llm = MagicMock()
        mock_logger = _make_logger()
        file_locks: dict = {}
        creating_pages: set = {_safe_slug("Duplicate Target")}
        creating_lock = asyncio.Lock()

        repaired, errors = await _repair_broken_link(
            cfg.wiki_dir, "Duplicate Target", ["page-a"],
            cfg, mock_llm, mock_logger,
            creating_pages, creating_lock, file_locks,
        )

        assert repaired == []
        assert errors == []
        mock_llm.call.assert_not_called()


# ---------------------------------------------------------------------------
# _repair_orphan
# ---------------------------------------------------------------------------


class TestRepairOrphan:
    @pytest.mark.asyncio
    async def test_returns_error_when_page_not_found(self, tmp_path):
        from src.stages.repair import _repair_orphan

        cfg = _make_cfg(tmp_path)
        mock_llm = MagicMock()
        mock_logger = _make_logger()
        file_locks: dict = {}

        repaired, errors = await _repair_orphan(
            cfg.wiki_dir, "nonexistent-page",
            cfg, mock_llm, mock_logger, file_locks,
        )

        assert repaired == []
        assert len(errors) == 1
        assert "nonexistent-page" in errors[0]
        assert "not found" in errors[0]

    @pytest.mark.asyncio
    async def test_adds_backlinks_when_llm_suggests_pages(self, tmp_path):
        from src.stages.repair import _repair_orphan

        cfg = _make_cfg(tmp_path)
        # Create orphan page
        orphan = cfg.wiki_dir / "orphan-page.md"
        orphan.write_text("# Orphan Page\n\nI have no backlinks.", encoding="utf-8")
        # Create a candidate page
        candidate = cfg.wiki_dir / "candidate.md"
        candidate.write_text("# Candidate\n\nRelated content.", encoding="utf-8")

        mock_llm = MagicMock()
        # LLM suggests 'candidate' as a page that should link to orphan
        mock_llm.call = AsyncMock(return_value=_make_llm_response("candidate page"))
        mock_logger = _make_logger()
        file_locks: dict = {}

        repaired, errors = await _repair_orphan(
            cfg.wiki_dir, "orphan-page",
            cfg, mock_llm, mock_logger, file_locks,
        )

        assert errors == []
        # candidate page should now contain a link to orphan-page
        assert "[[orphan-page]]" in candidate.read_text(encoding="utf-8")
        assert repaired == ["orphan-page"]

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_failure(self, tmp_path):
        from src.stages.repair import _repair_orphan

        cfg = _make_cfg(tmp_path)
        orphan = cfg.wiki_dir / "orphan.md"
        orphan.write_text("# Orphan\n\nContent.", encoding="utf-8")

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        mock_logger = _make_logger()
        file_locks: dict = {}

        repaired, errors = await _repair_orphan(
            cfg.wiki_dir, "orphan",
            cfg, mock_llm, mock_logger, file_locks,
        )

        assert repaired == []
        assert len(errors) == 1
        assert "orphan" in errors[0]


# ---------------------------------------------------------------------------
# run_repair
# ---------------------------------------------------------------------------


class TestRunRepair:
    @pytest.mark.asyncio
    async def test_returns_early_when_langgraph_not_installed(self, tmp_path):
        from src.stages.repair import run_repair

        cfg = _make_cfg(tmp_path)
        repair_state = RepairState(
            wiki_dir=cfg.wiki_dir,
            broken_links=[],
            orphans=[],
            evaluations=[],
        )
        mock_llm = MagicMock()
        mock_logger = _make_logger()

        with patch.dict("sys.modules", {"langgraph": None, "langgraph.graph": None, "langgraph.constants": None}):
            with patch("builtins.__import__", side_effect=ImportError("langgraph not installed")):
                result = await run_repair(repair_state, cfg, mock_llm, mock_logger)

        assert result is repair_state

    @pytest.mark.asyncio
    async def test_run_repair_empty_state(self, tmp_path):
        """Empty orphans/broken_links: graph runs and returns immediately."""
        from src.stages.repair import run_repair

        cfg = _make_cfg(tmp_path)
        repair_state = RepairState(
            wiki_dir=cfg.wiki_dir,
            broken_links=[],
            orphans=[],
            evaluations=[],
        )
        mock_llm = MagicMock()
        mock_logger = _make_logger()

        result = await run_repair(repair_state, cfg, mock_llm, mock_logger)

        assert result is repair_state
        assert result.repaired == []
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_run_repair_with_broken_link(self, tmp_path):
        """Broken link triggers stub page creation via LangGraph."""
        from src.stages.repair import run_repair

        cfg = _make_cfg(tmp_path)
        repair_state = RepairState(
            wiki_dir=cfg.wiki_dir,
            broken_links=[{"source": "page-a", "target": "Missing Topic"}],
            orphans=[],
            evaluations=[],
        )

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=_make_llm_response("Stub content for Missing Topic."))
        mock_logger = _make_logger()

        result = await run_repair(repair_state, cfg, mock_llm, mock_logger)

        assert "Missing Topic" in result.repaired
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_run_repair_with_orphan(self, tmp_path):
        """Orphan page triggers backlink insertion via LangGraph."""
        from src.stages.repair import run_repair

        cfg = _make_cfg(tmp_path)
        orphan = cfg.wiki_dir / "orphan-page.md"
        orphan.write_text("# Orphan Page\n\nNo backlinks.", encoding="utf-8")
        candidate = cfg.wiki_dir / "related.md"
        candidate.write_text("# Related\n\nSome content.", encoding="utf-8")

        repair_state = RepairState(
            wiki_dir=cfg.wiki_dir,
            broken_links=[],
            orphans=["orphan-page"],
            evaluations=[],
        )

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=_make_llm_response("related page should link"))
        mock_logger = _make_logger()

        result = await run_repair(repair_state, cfg, mock_llm, mock_logger)

        assert result.errors == []
