"""Filesystem reader — walks content_new/ and converts every supported file to a Document.

Supported formats:
  .md / .txt    — native (read as-is)
  .docx / .pptx / .xlsx — converted via markitdown
  .pdf          — converted via a pluggable PdfReaderProtocol (None = skip with warning)

The object_type for each document is resolved in priority order:
  1. ``object_type`` field in the YAML frontmatter
  2. Name of the immediate subdirectory inside content_dir (if it matches a known slug)
  3. Slug of the first ObjectTypeConfig (last-resort fallback)

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
import re
import uuid
from pathlib import Path

from markdown_hero import extract_frontmatter, remove_frontmatter, strip as md_strip

from ..models.config import WikiConfig
from ..models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)

_SUPPORTED = {".md", ".txt", ".doc", ".docx", ".pptx", ".xlsx", ".pdf"}


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")



def _humanize(stem: str) -> str:
    """Convert a filename stem into a human-readable title.

    Args:
        stem: The filename without extension (e.g. "my-document_v2").

    Returns:
        A title-cased string (e.g. "My Document V2").
    """
    return stem.replace("-", " ").replace("_", " ").title()


def _extract_md(raw: str, path: Path, cfg: WikiConfig, default_slug: str) -> Document:
    """Parse a raw Markdown string into a Document with a content-addressable ID.

    Args:
        raw: Full file content including any YAML frontmatter.
        path: Filesystem path of the source file.
        cfg: Active WikiConfig.
        default_slug: Object-type slug to use when no ``object_type`` field
            is present in the frontmatter.

    Returns:
        A Document with a populated DocumentMetadata and the original body as
        ``content``.
    """
    meta = extract_frontmatter(raw)
    body, _ = remove_frontmatter(raw)

    clean = md_strip(body)
    clean_bytes = clean.encode("utf-8")
    digest = hashlib.sha256(clean_bytes)
    doc_id = str(uuid.UUID(bytes=digest.digest()[:16]))

    title = str(meta.get("title") or _humanize(path.stem))
    object_type = str(meta.get("object_type") or default_slug)
    status = str(meta.get("status") or "")

    extra = {k: v for k, v in meta.items() if k not in {"id", "title", "object_type", "status"}}
    extra["content_sha256"] = digest.hexdigest()
    extra["source_filename"] = path.name

    return Document(
        metadata=DocumentMetadata(
            id=doc_id,
            title=title,
            object_type=object_type,
            status=status,
            extra=extra,
        ),
        content=body.strip(),
        content_path=path,
    )


def _read_file_sync(path: Path, cfg: WikiConfig) -> Document | None:
    """Synchronously read and convert a single file to a Document.

    Args:
        path: Path to the file to read.
        cfg: Active WikiConfig; provides object_type slugs and pdf_reader.

    Returns:
        A Document on success, or None if the file should be skipped.
    """
    suffix = path.suffix.lower()

    # Determine fallback object_type from parent folder name
    parent_slug = _slugify(path.parent.name) if path.parent != cfg.content_dir else ""
    # Validate parent_slug against known object types
    known_slugs = {obj.slug for obj in cfg.objects}
    default_slug = parent_slug if parent_slug in known_slugs else cfg.objects[0].slug

    try:
        if suffix in {".md", ".txt"}:
            raw = path.read_text(encoding="utf-8")
            return _extract_md(raw, path, cfg, default_slug)

        elif suffix in {".doc", ".docx", ".pptx", ".xlsx"}:
            try:
                from markitdown import MarkItDown  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "markitdown not installed — cannot process .doc/.docx/.pptx/.xlsx files"
                ) from exc
            md_converter = MarkItDown()
            result = md_converter.convert(str(path))
            raw = result.text_content or ""
            if not raw.strip():
                raise ValueError(f"No extractable text content in {path.name}")
            return _extract_md(raw, path, cfg, default_slug)

        elif suffix == ".pdf":
            if cfg.pdf_reader is None:
                logger.warning("pdf_reader not configured — skipping %s", path.name)
                return None
            raw = cfg.pdf_reader.extract_text(path)
            if not raw.strip():
                raise ValueError(
                    f"{path.name} contains no extractable text "
                    f"(scanned PDF without OCR support configured)"
                )
            return _extract_md(raw, path, cfg, default_slug)

    except ValueError:
        raise  # propagate content-quality errors to read_all
    except Exception as exc:  # noqa: BLE001
        logger.error("Error reading %s: %s", path, exc)
        raise


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
                ``objects`` provide slug validation.
        """
        self._cfg = cfg

    async def read_all(self) -> list[Document]:
        """Discover and convert every supported file under content_dir.

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
            try:
                doc = await asyncio.to_thread(_read_file_sync, path, self._cfg)
            except Exception as exc:  # noqa: BLE001
                logger.error("Cannot read %s: %s — skipping", path.name, exc)
                continue
            if doc is not None:
                docs.append(doc)

        logger.info("FilesystemReader: %d documents loaded from %s", len(docs), content_dir)
        return docs
