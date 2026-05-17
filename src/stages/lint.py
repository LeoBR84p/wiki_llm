"""Stage 6 — Static and semantic lint analysis.

1. static_analysis(): scans wiki_dir to find orphan pages and broken wikilinks
2. markdown_hero.lint() on every page: detects structural Markdown issues
3. LLM semantic lint: evaluates overall content quality using the lint prompt
4. Writes a consolidated lint_report.md to wiki_dir
5. Returns a RepairState for the repair stage to consume
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from ..llm.base import BaseLLMClient
from ..llm.log import LLMLogger
from ..models.config import WikiConfig
from ..models.evaluation import RepairState
from ._utils import SYSTEM_PAGES

logger = logging.getLogger(__name__)


def _extract_wikilinks(text: str) -> set[str]:
    return set(re.findall(r"\[\[([^\]]+)\]\]", text))


def _load_pages(wiki_dir: Path) -> dict[str, str]:
    pages: dict[str, str] = {}
    for p in wiki_dir.rglob("*.md"):
        if p.name in SYSTEM_PAGES or p.stem.startswith("lint_"):
            continue
        pages[p.stem] = p.read_text(encoding="utf-8")
    return pages


def static_analysis(wiki_dir: Path) -> dict[str, Any]:
    """Analyze the wiki for structural health issues without calling the LLM.

    Builds an inbound/outbound link graph from all non-system pages, then
    identifies orphan pages (pages with no inbound links) and broken links
    (wikilinks pointing to pages that do not exist on disk).

    Args:
        wiki_dir: Root wiki directory to analyze.

    Returns:
        A dict with keys:
          - ``orphans``: list of page stems with no inbound links
          - ``broken_links``: list of dicts with ``origem`` and ``destino`` keys
          - ``stats``: dict with ``total_paginas`` and ``total_links`` counts
    """
    if not wiki_dir.exists():
        return {"orphans": [], "broken_links": [], "stats": {"total_paginas": 0}}

    pages = _load_pages(wiki_dir)
    existing_ids = set(pages)

    outbound: dict[str, set[str]] = {pid: _extract_wikilinks(text) - {"index"} for pid, text in pages.items()}
    inbound: dict[str, set[str]] = {pid: set() for pid in existing_ids}

    for src, dests in outbound.items():
        for dst in dests:
            if dst in inbound:
                inbound[dst].add(src)

    index_path = wiki_dir / "index.md"
    if index_path.exists():
        for dst in _extract_wikilinks(index_path.read_text(encoding="utf-8")):
            if dst in inbound:
                inbound[dst].add("index")

    orphans = [pid for pid, inc in inbound.items() if not inc]
    broken: list[dict[str, str]] = [
        {"source": src, "target": dst}
        for src, dests in outbound.items()
        for dst in dests
        if dst not in existing_ids
    ]

    return {
        "orphans": orphans,
        "broken_links": broken,
        "stats": {
            "total_pages": len(pages),
            "total_links": sum(len(v) for v in outbound.values()),
        },
    }


async def run_lint(
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> RepairState:
    """Run the full lint stage: static analysis + markdown_hero lint + LLM semantic lint.

    Combines three layers of analysis:
    1. static_analysis() for link graph health (orphans, broken links)
    2. markdown_hero.lint() for structural Markdown quality per page
    3. LLM evaluation of overall wiki quality using the lint prompt

    Writes a comprehensive lint_report.md to wiki_dir and returns a RepairState
    so that the repair stage can immediately act on the detected issues.

    Args:
        cfg: Active WikiConfig with wiki_dir and prompt_lint path.
        llm: Active LLM client.
        llm_logger: Logger for the LLM semantic lint call.

    Returns:
        A RepairState populated with orphans and broken_links lists.
    """
    from markdown_hero import lint as mh_lint  # noqa: PLC0415

    # --- Static ---
    result = static_analysis(cfg.wiki_dir)
    orphans: list[str] = result["orphans"]
    broken_links: list[dict[str, str]] = result["broken_links"]
    logger.info("Static lint: %d orphans, %d broken links", len(orphans), len(broken_links))

    # --- markdown_hero lint ---
    mh_issues: list[str] = []
    pages = _load_pages(cfg.wiki_dir)
    for pid, text in pages.items():
        try:
            issues = mh_lint(text)
            if issues:
                for issue in issues:
                    mh_issues.append(f"{pid}: {issue}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("mh_lint failed for %s: %s", pid, exc)

    # --- LLM semantic lint ---
    system = cfg.prompt_lint.read_text(encoding="utf-8")
    stats = result["stats"]
    context_lines = [
        f"**Total pages:** {stats['total_pages']}",
        f"**Total links:** {stats['total_links']}",
        f"**Orphan pages ({len(orphans)}):** {', '.join(orphans[:30])}",
        f"**Broken links ({len(broken_links)}):** "
        + ", ".join(f"{b['source']}→{b['target']}" for b in broken_links[:20]),
    ]
    if mh_issues:
        context_lines.append(f"\n**Markdown issues ({len(mh_issues)}):**\n" + "\n".join(mh_issues[:40]))

    user = "\n".join(context_lines)
    t0 = llm_logger.start_call()
    llm_report = ""
    try:
        resp = await llm.call(system, user)
        llm_report = resp.text
        llm_logger.record(
            system=system, user=user, output=resp.text,
            tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            cached_tokens=resp.cached_tokens, model_id=resp.model_id,
            stage="lint.semantic", elapsed=time.monotonic() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM lint failed: %s", exc)

    # --- Write report ---
    report_lines = [
        "---",
        "type: lint_report",
        "---",
        "",
        "# Lint Report",
        "",
        "## Statistics",
        f"- Total pages: {stats['total_pages']}",
        f"- Total links: {stats['total_links']}",
        f"- Orphan pages: {len(orphans)}",
        f"- Broken links: {len(broken_links)}",
        "",
        "## Orphan Pages",
        "",
    ]
    for o in orphans:
        report_lines.append(f"- [[{o}]]")
    report_lines += [
        "",
        "## Broken Links",
        "",
    ]
    for b in broken_links:
        report_lines.append(f"- [[{b['origem']}]] → `{b['destino']}`")
    report_lines += [
        "",
        "## Semantic Evaluation (LLM)",
        "",
        llm_report or "_No LLM evaluation._",
    ]

    report_path = cfg.wiki_dir / "lint_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Lint report written: %s", report_path)

    return RepairState(
        wiki_dir=cfg.wiki_dir,
        orphans=orphans,
        broken_links=broken_links,
    )
