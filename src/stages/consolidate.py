"""Stage 5 — Semantic consolidation of duplicate wiki pages.

Flow:
1. markdown_hero.markdown_merge() for structural pre-pass deduplication
2. LLM identifies duplicate groups in batches of up to 80 pages
3. For each group: rename to canonical title, replace all [[wikilinks]] across the wiki

Only operates on object type subdirectories.  Existing pages are overwritten
only when their slug matches a detected duplicate.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from ..llm.base import BaseLLMClient
from ..llm.log import LLMLogger
from ..models.config import WikiConfig
from ._utils import SYSTEM_PAGES, collect_wiki_pages, write_atomic, _safe_slug

logger = logging.getLogger(__name__)


def _collect_pages(subdir: Path) -> list[dict[str, str]]:
    pages = []
    for p in collect_wiki_pages(subdir):
        text = p.read_text(encoding="utf-8")[:500]
        m = re.match(r"---\s*(.*?)\s*---", text, re.DOTALL)
        title = p.stem
        if m:
            tm = re.search(r"^title:\s*(.+)$", m.group(1), re.MULTILINE)
            if tm:
                title = tm.group(1).strip().strip('"\'')
        pages.append({"slug": p.stem, "name": title})
    return pages


def _replace_wikilinks(wiki_dir: Path, old_slug: str, new_slug: str) -> int:
    """Replace all occurrences of [[old_slug]] with [[new_slug]] across the wiki.

    Args:
        wiki_dir: Root wiki directory to scan recursively.
        old_slug: The slug being replaced (the duplicate).
        new_slug: The canonical slug to use instead.

    Returns:
        Number of files that were modified.
    """
    count = 0
    for p in wiki_dir.rglob("*.md"):
        if p.name in SYSTEM_PAGES:
            continue
        text = p.read_text(encoding="utf-8")
        updated = text.replace(f"[[{old_slug}]]", f"[[{new_slug}]]")
        if updated != text:
            p.write_text(updated, encoding="utf-8")
            count += 1
    return count


def _add_aliases(path: Path, aliases: list[str]) -> None:
    """Append alias strings to the ``aliases`` field in a page's YAML frontmatter.

    Args:
        path: Path to the Markdown file to update.
        aliases: List of alias strings to add (typically the duplicate titles).
    """
    if not path.exists() or not aliases:
        return
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return
    parts = text.split("---", 2)
    if len(parts) < 3:
        return
    fm, body = parts[1], parts[2]
    m = re.search(r"^aliases:\s*(.+)$", fm, re.MULTILINE)
    if m:
        try:
            existing: list[str] = json.loads(m.group(1).replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            existing = []
        all_aliases = list(dict.fromkeys(existing + aliases))
        fm = re.sub(
            r"^aliases:\s*.+$",
            f"aliases: {json.dumps(all_aliases, ensure_ascii=False)}",
            fm,
            flags=re.MULTILINE,
        )
    else:
        fm = fm.rstrip("\n") + f"\naliases: {json.dumps(aliases, ensure_ascii=False)}\n"
    write_atomic(path, f"---{fm}---{body}")


async def _identify_groups(
    pages: list[dict[str, str]],
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> list[dict[str, Any]]:
    """Ask the LLM to identify groups of semantically duplicate pages.

    Args:
        pages: List of dicts with ``slug`` and ``name`` keys from _collect_pages.
        cfg: Pipeline config providing the consolidate prompt and model settings.
        llm: Active LLM client.
        llm_logger: Logger for each consolidation call.

    Returns:
        A flat list of duplicate group dicts parsed from all batch responses.
    """
    system = cfg.prompt_consolidate.read_text(encoding="utf-8")
    names = [p["name"] for p in pages]
    groups: list[dict[str, Any]] = []

    for i in range(0, len(names), cfg.batch_size):
        batch = names[i: i + cfg.batch_size]
        user = "\n".join(f"- {n}" for n in batch)
        t0 = llm_logger.start_call()
        try:
            resp = await llm.call(system, user)
            llm_logger.record(
                system=system, user=user, output=resp.text,
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                cached_tokens=resp.cached_tokens, model_id=resp.model_id,
                stage="consolidate.identify", elapsed=time.monotonic() - t0,
            )
            text = resp.text.strip()
            m_fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            if m_fence:
                text = m_fence.group(1)
            else:
                m_arr = re.search(r"\[.*\]", text, re.DOTALL)
                text = m_arr.group() if m_arr else "[]"
            parsed = json.loads(text)
            groups.extend(
                g for g in parsed
                if g.get("canonical") and isinstance(g.get("duplicates"), list)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Consolidation batch %d failed: %s", i, exc)

    return groups


def _execute_merge(wiki_dir: Path, subdir_path: Path, group: dict[str, Any]) -> dict[str, Any]:
    """Merge a group of duplicate pages into a single canonical page.

    Args:
        wiki_dir: Root wiki directory (needed for cross-wiki link replacement).
        subdir_path: The object type subdirectory containing the duplicate files.
        group: A dict with ``canonical`` (str) and ``duplicates`` (list[str]) keys.

    Returns:
        A dict with ``canonical`` (str) and ``merged`` (list of merged titles).
    """
    canon_name = group["canonical"]
    canon_slug = _safe_slug(canon_name)
    canon_path = subdir_path / f"{canon_slug}.md"
    duplicates: list[str] = list(group.get("duplicates") or [])
    merged: list[str] = []
    canon_promoted = canon_path.exists()

    for dup_name in duplicates:
        dup_slug = _safe_slug(dup_name)
        if dup_slug == canon_slug:
            continue
        dup_path = subdir_path / f"{dup_slug}.md"
        if not canon_promoted and dup_path.exists():
            text = dup_path.read_text(encoding="utf-8")
            title_safe = canon_name.replace('"', "'")
            text = re.sub(r"^(title:\s*).*$", f'\\1"{title_safe}"', text, flags=re.MULTILINE)
            text = re.sub(r"^(# ).*", f"\\1{canon_name}", text, count=1, flags=re.MULTILINE)
            write_atomic(canon_path, text)
            canon_promoted = True
        _replace_wikilinks(wiki_dir, dup_slug, canon_slug)
        if dup_path.exists():
            dup_path.unlink(missing_ok=True)
            merged.append(dup_name)

    if merged:
        _add_aliases(canon_path, merged)

    return {"canonical": canon_name, "merged": merged}


async def run_consolidate(
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> None:
    """Run the full consolidation stage for all object type subdirectories.

    For each object type, runs a markdown_merge pre-pass to handle structural
    duplicates, then asks the LLM to identify semantic duplicates in batches,
    and finally executes the merge for each detected duplicate group.

    Args:
        cfg: Active WikiConfig with wiki_dir, objects, and consolidate prompt.
        llm: Active LLM client.
        llm_logger: Logger for all LLM calls made during this stage.
    """
    for obj in cfg.objects:
        subdir = cfg.wiki_dir / obj.wiki_subdir
        if not subdir.exists():
            continue

        pages = _collect_pages(subdir)
        if len(pages) < 2:
            continue

        logger.info("Consolidating %s: %d pages", obj.name, len(pages))
        groups = await _identify_groups(pages, cfg, llm, llm_logger)
        logger.info("  %d duplicate groups detected", len(groups))

        for group in groups:
            result = _execute_merge(cfg.wiki_dir, subdir, group)
            if result["merged"]:
                logger.info("  Merged %s → %s", result["merged"], result["canonical"])
