"""Shared utilities for pipeline stages."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

CHARS_INVALID: frozenset[str] = frozenset('\\/:*?"<>|')
SYSTEM_PAGES: frozenset[str] = frozenset({"index.md", "log.md", "lint_report.md"})


def write_atomic(path: Path, content: str, skip_if_exists: bool = False) -> bool:
    """Write content to path via a temp file then atomic rename.

    Returns True if written, False if skipped (skip_if_exists=True and file exists).
    Cleans up the temp file on any exception.
    """
    if skip_if_exists and path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f"._tmp_{uuid.uuid4().hex[:8]}" + path.suffix)
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _safe_slug(name: str) -> str:
    """Convert a page name to a filesystem-safe lowercase slug."""
    s = "".join(c if c not in CHARS_INVALID else "-" for c in name.lower().strip())
    return re.sub(r"-{2,}", "-", s).strip("-") or "page"


def collect_wiki_pages(
    directory: Path,
    skip: frozenset[str] = SYSTEM_PAGES,
) -> list[Path]:
    """Return sorted .md paths in *directory*, excluding system pages."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.md") if p.name not in skip)
