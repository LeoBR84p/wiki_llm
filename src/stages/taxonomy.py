"""Stage 2 — Taxonomy collection, normalization, and page generation.

1. Scan wiki_dir and collect raw terms from the configured section header
2. Normalize raw terms to canonical forms via LLM in batches
3. Generate or update taxonomy pages (one per canonical term)
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
from ..models.config import TaxonomyConfig, WikiConfig

logger = logging.getLogger(__name__)

_RE_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
_RE_MDLINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_CHARS_INVALID = frozenset('\\/:*?"<>|')
_SYSTEM_PAGES = {"index.md", "log.md"}


def _safe_slug(name: str) -> str:
    """Convert a term name to a filesystem-safe slug for use as a filename stem.

    Args:
        name: Human-readable term name (e.g. "Quality Management").

    Returns:
        A sanitized slug string (e.g. "Quality-Management").
    """
    s = "".join(c if c not in _CHARS_INVALID else "-" for name_char in name for c in [name_char])
    return s.strip(". ") or "tema-sem-nome"


def _section_pattern(header: str) -> re.Pattern:
    """Compile a regex that captures the body of a named Markdown section.

    The pattern matches the section from the given header line up to the next
    heading of any level, or the end of the document.

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


def collect_terms(wiki_dir: Path, tax_cfg: TaxonomyConfig) -> dict[str, list[str]]:
    """Scan wiki_dir and return a mapping of raw terms to the page IDs that mention them.

    Walks every non-system Markdown file under wiki_dir, extracts the section
    matching ``tax_cfg.section_header``, and collects all ``[[wikilinks]]``
    found inside that section.  The taxonomy subdirectory itself is excluded
    to avoid circular self-references.

    Args:
        wiki_dir: Root wiki directory to scan.
        tax_cfg: Taxonomy configuration providing the section_header and wiki_subdir.

    Returns:
        A dict mapping each raw term string to a list of page IDs (stems) that
        contain it under the configured section.
    """
    pattern = _section_pattern(tax_cfg.section_header)
    terms: dict[str, list[str]] = {}

    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name in _SYSTEM_PAGES:
            continue
        if tax_cfg.wiki_subdir in md_file.parts:
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


async def normalize_terms(
    terms: dict[str, list[str]],
    tax_cfg: TaxonomyConfig,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> dict[str, str]:
    """Normalize raw terms to canonical forms via LLM, processing in batches.

    Sends raw term names to the LLM in batches of cfg.batch_size and expects
    a JSON object mapping raw → canonical.  Falls back to an identity mapping
    (raw term = canonical term) for any batch that fails, ensuring the pipeline
    always produces output even when the LLM is unavailable.

    Args:
        terms: Raw terms to normalize (keys of the dict from collect_terms).
        tax_cfg: Provides the normalize prompt path.
        cfg: Pipeline config for batch_size.
        llm: Active LLM client.
        llm_logger: Logger to record each normalization call.

    Returns:
        A dict mapping each raw term to its canonical normalized form.
    """
    if not terms:
        return {}

    system_tpl = tax_cfg.prompt_normalize.read_text(encoding="utf-8")
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
                stage="taxonomy.normalize", elapsed=time.monotonic() - t0,
            )
            # Expected: JSON object or list {bruto: normalizado}
            bloco = re.search(r"\{.*\}", resp.text, re.DOTALL)
            if bloco:
                parsed = json.loads(bloco.group())
                mapping.update(parsed)
            else:
                # Fallback: identity mapping
                for n in batch:
                    mapping[n] = n
        except Exception as exc:  # noqa: BLE001
            logger.warning("normalize_terms batch %d failed: %s — using identity", i, exc)
            for n in batch:
                mapping[n] = n

    return mapping


async def generate_taxonomy_pages(
    terms: dict[str, list[str]],
    mapping: dict[str, str],
    tax_cfg: TaxonomyConfig,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> None:
    """Generate wiki pages for each canonical taxonomy term.

    Pivots the raw → canonical mapping to group page IDs under each canonical
    term, then calls the LLM to write a summary page for each term that does
    not already have one (incremental: existing pages are not overwritten).

    Args:
        terms: Raw term → [page_id, ...] mapping from collect_terms.
        mapping: Raw term → canonical term mapping from normalize_terms.
        tax_cfg: Provides the create_page prompt path and wiki_subdir.
        cfg: Pipeline config for wiki_dir.
        llm: Active LLM client.
        llm_logger: Logger to record each page-creation call.
    """
    import uuid as _uuid, time as _time  # noqa: E401

    def _write_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + f"._tmp_{_uuid.uuid4().hex[:8]}" + path.suffix)
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    # Pivot: normalizado → [page_ids]
    pivot: dict[str, list[str]] = {}
    for raw_term, page_ids in terms.items():
        normalized = mapping.get(raw_term, raw_term)
        bucket = pivot.setdefault(normalized, [])
        for pid in page_ids:
            if pid not in bucket:
                bucket.append(pid)

    system_tpl = tax_cfg.prompt_create_page.read_text(encoding="utf-8")
    tax_dir = cfg.wiki_dir / tax_cfg.wiki_subdir

    for norm_term, page_ids in pivot.items():
        slug = _safe_slug(norm_term)
        dest = tax_dir / f"{slug}.md"
        if dest.exists():
            continue  # incremental: do not overwrite

        links_parts = []
        for pid in sorted(page_ids):
            found = next((f for f in cfg.wiki_dir.rglob(f"{pid}.md") if f.stem == pid), None)
            if found:
                rel = os.path.relpath(found, tax_dir).replace("\\", "/")
                links_parts.append(f"[{pid}]({rel})")
            else:
                links_parts.append(f"[[{pid}]]")
        links = ", ".join(links_parts)
        context = {
            "term": norm_term,
            "page_ids": page_ids,
            "links": links,
            "taxonomy": tax_cfg.name,
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
                stage="taxonomy.create_page", elapsed=_time.monotonic() - t0,
            )
            _write_atomic(dest, resp.text)
            logger.info("Taxonomy page generated: %s", dest)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error generating taxonomy page '%s': %s", norm_term, exc)


async def run_taxonomy(
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> None:
    """Run the full taxonomy stage for all configured TaxonomyConfig entries.

    For each taxonomy in cfg.taxonomies, runs collect_terms → normalize_terms
    → generate_taxonomy_pages in sequence.  Logs a summary of raw term counts.
    Does nothing if no taxonomies are configured.

    Args:
        cfg: Active WikiConfig with wiki_dir and taxonomies list.
        llm: Active LLM client.
        llm_logger: Logger for all LLM calls made during this stage.
    """
    if not cfg.taxonomies:
        logger.info("No taxonomies configured.")
        return
    for tax_cfg in cfg.taxonomies:
        logger.info("Taxonomy: %s", tax_cfg.name)
        terms = collect_terms(cfg.wiki_dir, tax_cfg)
        logger.info("  %d raw terms collected", len(terms))
        mapping = await normalize_terms(terms, tax_cfg, cfg, llm, llm_logger)
        await generate_taxonomy_pages(terms, mapping, tax_cfg, cfg, llm, llm_logger)
