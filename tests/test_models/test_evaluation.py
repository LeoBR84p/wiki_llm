"""Unit tests for src/models/evaluation.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.evaluation import PageEvaluation, RepairState


class TestPageEvaluation:
    def test_approved_false_by_default(self):
        ev = PageEvaluation(approved=False)
        assert ev.approved is False
        assert ev.problems == []
        assert ev.suggestions == []

    def test_approved_true(self):
        ev = PageEvaluation(approved=True)
        assert ev.approved is True

    def test_with_problems_and_suggestions(self):
        ev = PageEvaluation(
            approved=False,
            problems=["Missing introduction", "No sources"],
            suggestions=["Add an intro paragraph", "Cite at least two sources"],
        )
        assert len(ev.problems) == 2
        assert "Missing introduction" in ev.problems
        assert len(ev.suggestions) == 2

    def test_serialization(self):
        ev = PageEvaluation(approved=True, problems=[], suggestions=["Keep it up"])
        d = ev.model_dump()
        assert d["approved"] is True
        assert d["suggestions"] == ["Keep it up"]


class TestRepairState:
    def test_minimal(self):
        rs = RepairState(wiki_dir="/tmp/wiki")
        assert isinstance(rs.wiki_dir, Path)
        assert rs.orphans == []
        assert rs.broken_links == []
        assert rs.repaired == []
        assert rs.errors == []

    def test_wiki_dir_accepts_path(self):
        rs = RepairState(wiki_dir=Path("/tmp/wiki"))
        assert isinstance(rs.wiki_dir, Path)

    def test_with_orphans_and_broken(self):
        rs = RepairState(
            wiki_dir="/wiki",
            orphans=["page-a", "page-b"],
            broken_links=[{"origem": "page-c", "destino": "page-x"}],
        )
        assert len(rs.orphans) == 2
        assert rs.broken_links[0]["origem"] == "page-c"

    def test_repaired_and_errors(self):
        rs = RepairState(
            wiki_dir="/wiki",
            repaired=["page-a"],
            errors=["Failed to fix page-b"],
        )
        assert "page-a" in rs.repaired
        assert "Failed to fix page-b" in rs.errors

    def test_serialization(self):
        rs = RepairState(wiki_dir="/wiki", orphans=["x"])
        d = rs.model_dump()
        assert isinstance(d["wiki_dir"], Path)
        assert d["orphans"] == ["x"]

    def test_wiki_dir_required(self):
        with pytest.raises(ValidationError):
            RepairState()
