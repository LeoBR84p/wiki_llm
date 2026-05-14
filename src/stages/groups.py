"""Stage 3 — Organizational grouping pages.

For each GroupingConfig, groups documents by the value of metadata.extra[metadata_field]
and generates one summary page per distinct value under wiki_dir/grouping.wiki_subdir/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from jinja2 import Template

from ..llm.base import BaseLLMClient
from ..llm.log import LLMLogger
from ..models.config import GroupingConfig, WikiConfig
from ..models.document import Document
from ._utils import CHARS_INVALID, write_atomic

logger = logging.getLogger(__name__)


def _safe_slug(name: str) -> str:
    s = "".join(c if c not in CHARS_INVALID else "-" for c in name)
    return s.strip(". ") or "unnamed-group"


def _group_value(doc: Document, field: str) -> str | None:
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
    lines = [
        "---",
        "type: grouping",
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


async def _generate_group_page_llm(
    group_name: str,
    grp_cfg: GroupingConfig,
    cfg: WikiConfig,
    docs: list[Document],
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> str:
    """Generate a grouping summary page via LLM.

    Renders the prompt template with group context, calls the LLM, and returns
    the generated Markdown content.  Falls back to the mechanical table if the
    LLM call fails.

    Args:
        group_name: The distinct metadata value defining this group.
        grp_cfg: GroupingConfig providing the prompt path.
        cfg: Active WikiConfig for language and wiki settings.
        docs: All documents belonging to this group.
        llm: Active LLM client.
        llm_logger: Logger for the LLM call.

    Returns:
        Markdown content string for the grouping page.
    """
    system_tpl = grp_cfg.prompt_create_page.read_text(encoding="utf-8")  # type: ignore[union-attr]
    docs_list = [
        {"id": d.metadata.id, "title": d.metadata.title, "status": d.metadata.status}
        for d in sorted(docs, key=lambda d: d.metadata.id)
    ]
    context = {
        "group_name": group_name,
        "field": grp_cfg.metadata_field,
        "grouping_name": grp_cfg.name,
        "docs": docs_list,
        "total": len(docs),
        "language": cfg.language,
    }
    system = Template(system_tpl).render(**context)
    user = f"{grp_cfg.name}: {group_name}\nDocuments: {len(docs)}"

    t0 = llm_logger.start_call()
    try:
        resp = await llm.call(system, user)
        llm_logger.record(
            system=system, user=user, output=resp.text,
            tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            cached_tokens=resp.cached_tokens, model_id=resp.model_id,
            stage="groups.create_page", elapsed=time.monotonic() - t0,
        )
        return resp.text
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM generation for group '%s' failed: %s — using fallback table", group_name, exc)
        return _page_content(group_name, grp_cfg.metadata_field, docs, grp_cfg)


async def run_groups(
    cfg: WikiConfig,
    docs: list[Document],
    llm: BaseLLMClient | None = None,
    llm_logger: LLMLogger | None = None,
) -> None:
    """Generate grouping pages for all configured GroupingConfig entries.

    For each grouping, partitions the provided documents by their metadata
    field value, then writes one summary page per distinct value.  When
    ``grp_cfg.prompt_create_page`` is set and an LLM client is available the
    page is generated via LLM; otherwise a mechanical Markdown table is
    written.  Existing pages are not overwritten (incremental).  Does nothing
    if no groupings are configured.

    Args:
        cfg: Active WikiConfig with wiki_dir and groupings list.
        docs: List of Documents produced by the read stage.
        llm: Optional LLM client used for prompt-based page generation.
        llm_logger: Optional logger for LLM calls.
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
            use_llm = grp_cfg.prompt_create_page is not None and llm is not None and llm_logger is not None
            if use_llm:
                content = await _generate_group_page_llm(
                    group_name, grp_cfg, cfg, group_docs, llm, llm_logger  # type: ignore[arg-type]
                )
            else:
                content = _page_content(group_name, grp_cfg.metadata_field, group_docs, grp_cfg)
            write_atomic(dest, content)
            logger.info("  Grouping page generated: %s", dest.name)
