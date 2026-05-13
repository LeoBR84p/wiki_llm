"""Stage 3 — Organizational grouping pages.

For each GroupingConfig, groups documents by the value of metadata.extra[metadata_field]
and generates one summary page per distinct value under wiki_dir/grouping.wiki_subdir/.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from ..models.config import GroupingConfig, WikiConfig
from ..models.document import Document

logger = logging.getLogger(__name__)

_CHARS_INVALID = frozenset('\\/:*?"<>|')


def _safe_slug(name: str) -> str:
    """Convert a group value to a filesystem-safe slug for use as a page filename stem.

    Args:
        name: The raw metadata field value (e.g. a team name or category).

    Returns:
        A sanitized slug string safe for use as a filename.
    """
    s = "".join(c if c not in _CHARS_INVALID else "-" for c in name)
    return s.strip(". ") or "grupo-sem-nome"


def _write_atomic(path: Path, content: str) -> None:
    """Write content to path via a temporary file, then atomically rename.

    Args:
        path: Destination file path.
        content: UTF-8 text to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f"._tmp_{uuid.uuid4().hex[:8]}" + path.suffix)
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _group_value(doc: Document, field: str) -> str | None:
    """Extract the grouping field value from a Document.

    Looks up ``field`` in metadata.extra first, then falls back to a direct
    attribute on DocumentMetadata (e.g. ``status``).  Returns None if the
    field is absent or empty, so the document is excluded from grouping.

    Args:
        doc: The Document to inspect.
        field: The metadata field name to retrieve.

    Returns:
        The field value as a stripped string, or None if not present.
    """
    val = doc.metadata.extra.get(field) or getattr(doc.metadata, field, None)
    if val:
        return str(val).strip()
    return None


def _page_content(group_name: str, field: str, docs: list[Document], grp_cfg: GroupingConfig) -> str:
    """Build the Markdown content for a grouping summary page.

    Creates a frontmatter block and a table listing every document in the
    group with id, title, and status.  Documents are sorted by ID for
    deterministic output.

    Args:
        group_name: The distinct value that defines this group (e.g. "Finance Team").
        field: The metadata field name used for grouping.
        docs: All documents belonging to this group.
        grp_cfg: GroupingConfig that provides the section name.

    Returns:
        A complete Markdown string ready to be written to disk.
    """
    slug = _safe_slug(group_name)
    links = ", ".join(f"[[{d.metadata.id}]]" for d in sorted(docs, key=lambda d: d.metadata.id))
    lines = [
        "---",
        f'tipo: grouping',
        f'field: {field}',
        f'value: "{group_name}"',
        f"total: {len(docs)}",
        "---",
        "",
        f"# {grp_cfg.name}: {group_name}",
        "",
        f"Groups **{len(docs)}** document(s) with `{field} = {group_name}`.",
        "",
        "## Documents",
        "",
        "| id | title | status |",
        "|---|---|---|",
    ]
    for d in sorted(docs, key=lambda d: d.metadata.id):
        lines.append(f"| [[{d.metadata.id}]] | {d.metadata.title[:70]} | {d.metadata.status} |")

    return "\n".join(lines) + "\n"


async def run_groups(cfg: WikiConfig, docs: list[Document]) -> None:
    """Generate grouping pages for all configured GroupingConfig entries.

    For each grouping, partitions the provided documents by their metadata
    field value, then writes one summary page per distinct value.  Existing
    pages are not overwritten (incremental).  Does nothing if no groupings
    are configured.

    Args:
        cfg: Active WikiConfig with wiki_dir and groupings list.
        docs: List of Documents produced by the read stage.
    """
    if not cfg.groupings:
        return

    for grp_cfg in cfg.groupings:
        logger.info("Grouping: %s (field: %s)", grp_cfg.name, grp_cfg.metadata_field)
        groups: dict[str, list[Document]] = {}
        for doc in docs:
            val = _group_value(doc, grp_cfg.metadata_field)
            if val:
                groups.setdefault(val, []).append(doc)

        grp_dir = cfg.wiki_dir / grp_cfg.wiki_subdir
        for group_name, group_docs in groups.items():
            slug = _safe_slug(group_name)
            dest = grp_dir / f"{slug}.md"
            if dest.exists():
                continue
            content = _page_content(group_name, grp_cfg.metadata_field, group_docs, grp_cfg)
            _write_atomic(dest, content)
            logger.info("  Grouping page generated: %s", dest.name)
