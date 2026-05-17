"""Stage 2 — Key theme collection, normalization, and page generation.

1. Scan wiki_dir and collect raw terms from the configured section header
2. Normalize raw terms to canonical forms via LLM in batches
3. Generate or update key theme pages (one per canonical term)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from jinja2 import Template

from ..llm.base import BaseLLMClient
from ..llm.log import LLMLogger
from ..models.config import KeyThemeConfig, WikiConfig
from ._utils import SYSTEM_PAGES, write_atomic, _safe_slug

logger = logging.getLogger(__name__)

_RE_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
_RE_MDLINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _section_pattern(header: str) -> re.Pattern:
    """Compile a regex that captures the body of a named Markdown section.

    Args:
        header: Section heading text to search for (e.g. "## Topics").

    Returns:
        A compiled re.Pattern for use with re.search.
    """
    escaped = re.escape(header.lstrip("#").strip())
    return re.compile(
        r"#{1,6}\s+" + escaped + r"(.*?)(?=\n#{1,6}|\Z)",
        re.DOTALL | re.IGNORECASE,
    )


def _collect_terms_from_sections(wiki_dir: Path, theme_cfg: KeyThemeConfig) -> dict[str, list[str]]:
    """Collect terms by scanning ``[[wikilinks]]`` inside a named Markdown section.

    Args:
        wiki_dir: Root wiki directory to scan.
        theme_cfg: Key theme configuration providing the section_header and wiki_subdir.

    Returns:
        A dict mapping each raw term string to a list of page IDs (stems) that
        contain it under the configured section.
    """
    pattern = _section_pattern(theme_cfg.section_header)  # type: ignore[arg-type]
    terms: dict[str, list[str]] = {}

    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name in SYSTEM_PAGES:
            continue
        if theme_cfg.wiki_subdir in md_file.parts:
            continue
        text = md_file.read_text(encoding="utf-8")
        m = pattern.search(text)
        if not m:
            continue
        page_id = md_file.stem
        section_text = m.group(1)
        raw_terms = _RE_WIKILINK.findall(section_text) + _RE_MDLINK.findall(section_text)
        for term in raw_terms:
            t = term.strip()
            if t:
                terms.setdefault(t, [])
                if page_id not in terms[t]:
                    terms[t].append(page_id)

    return terms


_RE_FM_SCALAR = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<val>.+)$")
_RE_FM_LIST_ITEM = re.compile(r"^\s+-\s+(?P<val>.+)$")


def _collect_terms_from_frontmatter(wiki_dir: Path, theme_cfg: KeyThemeConfig) -> dict[str, list[str]]:
    """Collect terms from a frontmatter field in every wiki page.

    Args:
        wiki_dir: Root wiki directory to scan.
        theme_cfg: Key theme configuration providing the metadata_field and wiki_subdir.

    Returns:
        A dict mapping each term string to a list of page IDs (stems) that
        carry that value in their frontmatter.
    """
    field = theme_cfg.metadata_field  # type: ignore[assignment]
    terms: dict[str, list[str]] = {}

    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name in SYSTEM_PAGES:
            continue
        if theme_cfg.wiki_subdir in md_file.parts:
            continue
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        if end == -1:
            continue
        fm_text = text[4:end]
        page_id = md_file.stem

        in_target = False
        collected: list[str] = []
        for line in fm_text.splitlines():
            list_m = _RE_FM_LIST_ITEM.match(line) if in_target else None
            if list_m:
                val = list_m.group("val").strip().strip("\"'")
                if val:
                    collected.append(val)
                continue
            scalar_m = _RE_FM_SCALAR.match(line)
            if scalar_m:
                in_target = scalar_m.group("key") == field
                if in_target:
                    val = scalar_m.group("val").strip().strip("\"'")
                    if val.startswith("[") and val.endswith("]"):
                        for item in val[1:-1].split(","):
                            v = item.strip().strip("\"'")
                            if v:
                                collected.append(v)
                        in_target = False
                    elif val:
                        collected.append(val)
                        in_target = False

        for term in collected:
            terms.setdefault(term, [])
            if page_id not in terms[term]:
                terms[term].append(page_id)

    return terms


def collect_terms(wiki_dir: Path, theme_cfg: KeyThemeConfig) -> dict[str, list[str]]:
    """Dispatch term collection based on ``theme_cfg.term_source``.

    Args:
        wiki_dir: Root wiki directory to scan.
        theme_cfg: Key theme configuration.

    Returns:
        A dict mapping each raw term string to a list of page IDs.
    """
    if theme_cfg.term_source == "metadata_field":
        return _collect_terms_from_frontmatter(wiki_dir, theme_cfg)
    return _collect_terms_from_sections(wiki_dir, theme_cfg)


async def normalize_terms(
    terms: dict[str, list[str]],
    theme_cfg: KeyThemeConfig,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> dict[str, str]:
    """Normalize raw terms to canonical forms via LLM, processing in batches.

    Args:
        terms: Raw terms to normalize (keys of the dict from collect_terms).
        theme_cfg: Provides the normalize prompt path.
        cfg: Pipeline config for batch_size.
        llm: Active LLM client.
        llm_logger: Logger to record each normalization call.

    Returns:
        A dict mapping each raw term to its canonical normalized form.
    """
    if not terms:
        return {}

    system_tpl = theme_cfg.prompt_normalize.read_text(encoding="utf-8")
    names = list(terms.keys())
    batch_size = cfg.batch_size
    mapping: dict[str, str] = {}

    for i in range(0, len(names), batch_size):
        batch = names[i: i + batch_size]
        user = "\n".join(f"- {n}" for n in batch)
        system = Template(system_tpl).render(language=cfg.language)

        t0 = llm_logger.start_call()
        try:
            resp = await llm.call(system, user)
            llm_logger.record(
                system=system, user=user, output=resp.text,
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                cached_tokens=resp.cached_tokens, model_id=resp.model_id,
                stage="key_theme.normalize", elapsed=time.monotonic() - t0,
            )
            match = re.search(r"\{.*\}", resp.text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                mapping.update(parsed)
            else:
                for n in batch:
                    mapping[n] = n
        except Exception as exc:  # noqa: BLE001
            logger.warning("normalize_terms batch %d failed: %s — using identity", i, exc)
            for n in batch:
                mapping[n] = n

    return mapping


async def generate_key_theme_pages(
    terms: dict[str, list[str]],
    mapping: dict[str, str],
    theme_cfg: KeyThemeConfig,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> None:
    """Generate wiki pages for each canonical key theme term.

    Args:
        terms: Raw term → [page_id, ...] mapping from collect_terms.
        mapping: Raw term → canonical term mapping from normalize_terms.
        theme_cfg: Provides the create_page prompt path and wiki_subdir.
        cfg: Pipeline config for wiki_dir.
        llm: Active LLM client.
        llm_logger: Logger to record each page-creation call.
    """
    pivot: dict[str, list[str]] = {}
    for raw_term, page_ids in terms.items():
        normalized = mapping.get(raw_term, raw_term)
        bucket = pivot.setdefault(normalized, [])
        for pid in page_ids:
            if pid not in bucket:
                bucket.append(pid)

    system_tpl = theme_cfg.prompt_create_page.read_text(encoding="utf-8")
    theme_dir = cfg.wiki_dir / theme_cfg.wiki_subdir

    for norm_term, page_ids in pivot.items():
        slug = _safe_slug(norm_term)
        dest = theme_dir / f"{slug}.md"
        if dest.exists():
            continue

        links_parts = []
        for pid in sorted(page_ids):
            found = next((f for f in cfg.wiki_dir.rglob(f"{pid}.md") if f.stem == pid), None)
            if found:
                rel = os.path.relpath(found, theme_dir).replace("\\", "/")
                links_parts.append(f"[{pid}]({rel})")
            else:
                links_parts.append(f"[[{pid}]]")
        links = ", ".join(links_parts)
        context = {
            "term": norm_term,
            "page_ids": page_ids,
            "links": links,
            "key_theme": theme_cfg.name,
            "language": cfg.language,
        }
        system = Template(system_tpl).render(**context)
        user = f"Term: {norm_term}\nPages: {links}"

        t0 = llm_logger.start_call()
        try:
            resp = await llm.call(system, user)
            llm_logger.record(
                system=system, user=user, output=resp.text,
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                cached_tokens=resp.cached_tokens, model_id=resp.model_id,
                stage="key_theme.create_page", elapsed=time.monotonic() - t0,
            )
            write_atomic(dest, resp.text)
            logger.info("Key theme page generated: %s", dest)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error generating key theme page '%s': %s", norm_term, exc)


async def run_key_themes(
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> None:
    """Run the full key theme stage for all configured KeyThemeConfig entries.

    For each key theme in cfg.key_themes, runs collect_terms → normalize_terms
    → generate_key_theme_pages in sequence.  Does nothing if no key themes are configured.

    Args:
        cfg: Active WikiConfig with wiki_dir and key_themes list.
        llm: Active LLM client.
        llm_logger: Logger for all LLM calls made during this stage.
    """
    if not cfg.key_themes:
        logger.info("No key themes configured.")
        return
    for theme_cfg in cfg.key_themes:
        logger.info("Key theme: %s", theme_cfg.name)
        terms = collect_terms(cfg.wiki_dir, theme_cfg)
        logger.info("  %d raw terms collected", len(terms))
        mapping = await normalize_terms(terms, theme_cfg, cfg, llm, llm_logger)
        await generate_key_theme_pages(terms, mapping, theme_cfg, cfg, llm, llm_logger)
