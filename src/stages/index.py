"""Stage 4 — Global wiki index generation.

Generates wiki_dir/index.md with:
- One section per entity_type listing all its pages
- One section per taxonomy (if configured)
- One section per grouping (if configured)
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models.config import WikiConfig

logger = logging.getLogger(__name__)

_SYSTEM_FILES = {"index.md"}


def _collect_pages(directory: Path) -> list[Path]:
    """Return a sorted list of Markdown files in a directory, excluding system files.

    Args:
        directory: The directory to list.  Returns an empty list if it does not exist.

    Returns:
        Sorted list of .md Path objects, excluding index.md.
    """
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.glob("*.md") if p.name not in _SYSTEM_FILES
    )


async def run_index(cfg: WikiConfig) -> None:
    """Rebuild index.md from the current state of the wiki directory.

    Scans each entity_type, taxonomy, and grouping subdirectory for Markdown
    files and writes a fresh index.md.  This function is idempotent: running
    it multiple times produces the same result.

    Args:
        cfg: Active WikiConfig with wiki_dir, entity_types, taxonomies, and groupings.
    """
    lines: list[str] = [
        "---",
        "tipo: indice",
        "---",
        "",
        "# Wiki Index",
        "",
    ]

    for et in cfg.entity_types:
        subdir = cfg.wiki_dir / et.wiki_subdir
        pages = _collect_pages(subdir)
        if not pages:
            continue
        lines.append(f"## {et.name}")
        lines.append("")
        for p in pages:
            stem = p.stem
            lines.append(f"- [[{stem}]]")
        lines.append("")

    if cfg.taxonomies:
        lines.append("## Taxonomias")
        lines.append("")
        for tax in cfg.taxonomies:
            subdir = cfg.wiki_dir / tax.wiki_subdir
            pages = _collect_pages(subdir)
            lines.append(f"### {tax.name}")
            lines.append("")
            for p in pages:
                lines.append(f"- [[{p.stem}]]")
            lines.append("")

    if cfg.groupings:
        lines.append("## Agrupamentos")
        lines.append("")
        for grp in cfg.groupings:
            subdir = cfg.wiki_dir / grp.wiki_subdir
            pages = _collect_pages(subdir)
            lines.append(f"### {grp.name}")
            lines.append("")
            for p in pages:
                lines.append(f"- [[{p.stem}]]")
            lines.append("")

    dest = cfg.wiki_dir / "index.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Index generated: %s", dest)
