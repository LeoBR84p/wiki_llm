"""Filesystem reader — walks content_new/ and converts every supported file to a Document.

Supported formats:
  .md / .txt    — native (read as-is)
  .docx / .pptx / .xlsx — converted via markitdown
  .pdf          — converted via a pluggable PdfReaderProtocol (None = skip with warning)

The entity_type for each document is resolved in priority order:
  1. ``entity_type`` field in the YAML frontmatter
  2. Name of the immediate subdirectory inside content_dir (if it matches a known slug)
  3. Slug of the first EntityTypeConfig (last-resort fallback)

Document ID:
  A UUID derived from the first 128 bits of SHA-256 of the stripped text
  (markdown_hero.strip() applied to the body, without frontmatter).
  Deterministic: same content → same UUID regardless of superficial formatting.
  The raw page is written with the original, unstripped content.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path

from markdown_hero import extract_frontmatter, remove_frontmatter, strip as md_strip

from ..models.config import WikiConfig
from ..models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)

_SUPPORTED = {".md", ".txt", ".docx", ".pptx", ".xlsx", ".pdf"}


def _content_uuid(clean_text: str) -> str:
    """Derive a deterministic UUID from the SHA-256 of the stripped document body.

    Uses the first 16 bytes of the SHA-256 digest to construct a UUID, giving
    a content-addressable identifier that is stable across filename renames,
    metadata edits, and re-ingestion of identical content.  Invariant to
    superficial formatting differences because md_strip() is applied first.

    Args:
        clean_text: Stripped Markdown body (output of markdown_hero.strip()).

    Returns:
        A UUID string in canonical hyphenated form (e.g. "6ba7b810-9dad-...").
    """
    digest = hashlib.sha256(clean_text.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def _humanize(stem: str) -> str:
    """Convert a filename stem into a human-readable title.

    Replaces hyphens and underscores with spaces and title-cases the result.
    Used as the fallback title when no ``title`` field exists in frontmatter.

    Args:
        stem: The filename without extension (e.g. "my-document_v2").

    Returns:
        A title-cased string (e.g. "My Document V2").
    """
    return stem.replace("-", " ").replace("_", " ").title()


def _extract_md(raw: str, path: Path, cfg: WikiConfig, default_slug: str) -> Document:
    """Parse a raw Markdown string into a Document with a content-addressable ID.

    Splits frontmatter from body, strips the body for stable hashing, then
    assembles a Document whose id is a UUID derived from the content hash.
    The ``content`` field on the returned Document retains the original
    (unstripped) body so that the raw wiki page faithfully preserves the
    source text.

    Args:
        raw: Full file content including any YAML frontmatter.
        path: Filesystem path of the source file (used for fallback title and
            to populate source_filename in extras).
        cfg: Active WikiConfig (not used directly here but passed for future
            extensibility).
        default_slug: Entity-type slug to use when no ``entity_type`` field
            is present in the frontmatter.

    Returns:
        A Document with a populated DocumentMetadata and the original body as
        ``content``.
    """
    meta = extract_frontmatter(raw)
    body, _ = remove_frontmatter(raw)

    # Hash sobre texto limpo — invariante a formatação superficial (espaços, pontuação, acentos)
    clean = md_strip(body)
    doc_id = _content_uuid(clean)

    # Descriptive fields: read from frontmatter, fallback to filename
    title = str(meta.get("title") or meta.get("titulo") or _humanize(path.stem))
    entity_type = str(meta.get("entity_type") or default_slug)
    status = str(meta.get("status") or "")

    # Metadados extras: apenas campos do domínio, sem source_id
    extra = {k: v for k, v in meta.items() if k not in {"id", "norma_id", "title", "titulo", "entity_type", "status"}}
    extra["content_sha256"] = hashlib.sha256(clean.encode("utf-8")).hexdigest()
    extra["source_filename"] = path.name

    return Document(
        metadata=DocumentMetadata(
            id=doc_id,
            title=title,
            entity_type=entity_type,
            status=status,
            extra=extra,
        ),
        content=body.strip(),   # original content preserved for the raw page
        content_path=path,
    )


def _read_file_sync(path: Path, cfg: WikiConfig) -> Document | None:
    """Synchronously read and convert a single file to a Document.

    Dispatches to the correct conversion strategy based on file extension.
    Returns None (with a logged warning) if the format requires an optional
    dependency that is not installed, or if any exception occurs during
    reading.  Designed to be called via asyncio.to_thread so that blocking
    I/O and markitdown conversion do not block the event loop.

    Args:
        path: Path to the file to read.
        cfg: Active WikiConfig; provides entity_type slugs and pdf_reader.

    Returns:
        A Document on success, or None if the file should be skipped.
    """
    suffix = path.suffix.lower()

    # Determine fallback entity_type from parent folder name
    parent_slug = _slugify(path.parent.name) if path.parent != cfg.content_dir else ""
    # Validate parent_slug against known entity types
    known_slugs = {et.slug for et in cfg.entity_types}
    default_slug = parent_slug if parent_slug in known_slugs else cfg.entity_types[0].slug

    try:
        if suffix in {".md", ".txt"}:
            raw = path.read_text(encoding="utf-8")
            return _extract_md(raw, path, cfg, default_slug)

        elif suffix in {".docx", ".pptx", ".xlsx"}:
            try:
                from markitdown import MarkItDown  # noqa: PLC0415
                md_converter = MarkItDown()
                result = md_converter.convert(str(path))
                raw = result.text_content or ""
            except ImportError:
                logger.warning("markitdown not installed — skipping %s", path.name)
                return None
            return _extract_md(raw, path, cfg, default_slug)

        elif suffix == ".pdf":
            if cfg.pdf_reader is None:
                logger.warning("pdf_reader not configured — skipping %s", path.name)
                return None
            raw = cfg.pdf_reader.extract_text(path)
            return _extract_md(raw, path, cfg, default_slug)

    except Exception as exc:  # noqa: BLE001
        logger.error("Error reading %s: %s", path, exc)
        return None

    return None


class FilesystemReader:
    """Async reader that discovers and converts all documents under content_dir.

    Walks the content directory recursively, filters for supported extensions,
    and converts each file to a Document using asyncio.to_thread to avoid
    blocking the event loop during disk I/O and format conversion.
    """

    def __init__(self, cfg: WikiConfig) -> None:
        """Initialize the reader with the active pipeline configuration.

        Args:
            cfg: WikiConfig whose ``content_dir`` will be walked and whose
                ``entity_types`` provide slug validation.
        """
        self._cfg = cfg

    async def read_all(self) -> list[Document]:
        """Discover and convert every supported file under content_dir.

        Walks the directory tree recursively and processes each file in a
        thread pool via asyncio.to_thread.  Files that fail conversion are
        silently skipped (an error is logged per file).  The returned list
        contains only successfully converted Documents.

        Returns:
            A list of Document instances, one per successfully converted file.
        """
        content_dir = self._cfg.content_dir
        if not content_dir.exists():
            logger.warning("content_dir does not exist: %s", content_dir)
            return []

        paths = [p for p in content_dir.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED]
        docs: list[Document] = []

        for path in paths:
            doc = await asyncio.to_thread(_read_file_sync, path, self._cfg)
            if doc is not None:
                docs.append(doc)

        logger.info("FilesystemReader: %d documents loaded from %s", len(docs), content_dir)
        return docs
