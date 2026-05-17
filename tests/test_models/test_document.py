"""Unit tests for src/models/document.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.document import Document, DocumentMetadata


class TestDocumentMetadata:
    def test_minimal(self):
        meta = DocumentMetadata(
            id="abc123",
            title="Test Title",
            object_type="article",
        )
        assert meta.id == "abc123"
        assert meta.title == "Test Title"
        assert meta.object_type == "article"
        assert meta.status == ""
        assert meta.extra == {}

    def test_with_all_fields(self):
        meta = DocumentMetadata(
            id="def456",
            title="Full Doc",
            object_type="policy",
            status="active",
            extra={"author": "Alice", "year": 2024},
        )
        assert meta.status == "active"
        assert meta.extra["author"] == "Alice"

    def test_extra_defaults_to_empty_dict(self):
        meta = DocumentMetadata(id="x", title="X", object_type="y")
        assert isinstance(meta.extra, dict)
        assert len(meta.extra) == 0

    def test_serialization(self):
        meta = DocumentMetadata(id="z", title="Z", object_type="article", status="draft")
        d = meta.model_dump()
        assert d["id"] == "z"
        assert d["status"] == "draft"


class TestDocument:
    def test_minimal_no_path(self):
        doc = Document(
            metadata=DocumentMetadata(id="1", title="T", object_type="article"),
            content="# Title\n\nBody.",
        )
        assert doc.content_path is None
        assert doc.content.startswith("# Title")

    def test_with_path(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("content", encoding="utf-8")
        doc = Document(
            metadata=DocumentMetadata(id="2", title="T2", object_type="article"),
            content="content",
            content_path=p,
        )
        assert doc.content_path == p
        assert doc.content_path.exists()

    def test_metadata_access(self):
        doc = Document(
            metadata=DocumentMetadata(
                id="3",
                title="Doc Three",
                object_type="report",
                status="final",
                extra={"dept": "Finance"},
            ),
            content="body",
        )
        assert doc.metadata.title == "Doc Three"
        assert doc.metadata.extra["dept"] == "Finance"

    def test_empty_content(self):
        doc = Document(
            metadata=DocumentMetadata(id="4", title="Empty", object_type="note"),
            content="",
        )
        assert doc.content == ""

    def test_serialization_roundtrip(self):
        doc = Document(
            metadata=DocumentMetadata(id="5", title="RT", object_type="article"),
            content="roundtrip",
        )
        data = doc.model_dump()
        assert data["metadata"]["id"] == "5"
        assert data["content"] == "roundtrip"
        assert data["content_path"] is None
