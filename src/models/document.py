"""Document models that flow through the pipeline.

Documents are created by the reader stage and consumed by all subsequent
stages.  They carry both the raw content and the metadata needed to generate
wiki pages and maintain provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing a single ingested document.

    Attributes:
        id: Deterministic UUID string derived from the document content hash.
        title: Human-readable document title (from frontmatter or filename).
        object_type: Slug of the ObjectTypeConfig this document belongs to.
        status: Optional workflow status string (e.g. ``"active"``, ``"draft"``).
        extra: Arbitrary additional metadata fields from the document frontmatter.
    """
    id: str
    title: str
    object_type: str
    status: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A single ingested document ready for pipeline processing.

    Attributes:
        metadata: Structured metadata including ID, title, and object type.
        content: The full document body text (raw Markdown, stripped of frontmatter).
        content_path: Optional path to the original source file on disk.
    """
    metadata: DocumentMetadata
    content: str
    content_path: Path | None = None
