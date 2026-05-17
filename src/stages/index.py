"""Stage 4 — Global wiki index generation.

Generates wiki_dir/index.md with:
- One section per object type listing all its pages
- One section per key theme (if configured)
- One section per group (if configured)
"""

from __future__ import annotations

import logging

from ..models.config import WikiConfig
from ._utils import collect_wiki_pages

logger = logging.getLogger(__name__)


async def run_index(cfg: WikiConfig) -> None:
    """Rebuild index.md from the current state of the wiki directory.

    Scans each object type, key theme, and group subdirectory for Markdown
    files and writes a fresh index.md.  This function is idempotent.

    Args:
        cfg: Active WikiConfig with wiki_dir, objects, key_themes, and groups.
    """
    lines: list[str] = [
        "---",
        "type: index",
        "---",
        "",
        "# Wiki Index",
        "",
    ]

    for obj in cfg.objects:
        subdir = cfg.wiki_dir / obj.wiki_subdir
        pages = collect_wiki_pages(subdir)
        if not pages:
            continue
        lines.append(f"## {obj.name}")
        lines.append("")
        for p in pages:
            stem = p.stem
            lines.append(f"- [[{stem}]]")
        lines.append("")

    if cfg.key_themes:
        lines.append("## Key Themes")
        lines.append("")
        for theme in cfg.key_themes:
            subdir = cfg.wiki_dir / theme.wiki_subdir
            pages = collect_wiki_pages(subdir)
            lines.append(f"### {theme.name}")
            lines.append("")
            for p in pages:
                lines.append(f"- [[{p.stem}]]")
            lines.append("")

    if cfg.groups:
        lines.append("## Groups")
        lines.append("")
        for grp in cfg.groups:
            subdir = cfg.wiki_dir / grp.wiki_subdir
            pages = collect_wiki_pages(subdir)
            lines.append(f"### {grp.name}")
            lines.append("")
            for p in pages:
                lines.append(f"- [[{p.stem}]]")
            lines.append("")

    dest = cfg.wiki_dir / "index.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Index generated: %s", dest)
