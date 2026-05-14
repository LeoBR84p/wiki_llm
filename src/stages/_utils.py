"""Shared utilities for pipeline stages."""

from __future__ import annotations

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
