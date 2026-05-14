"""Stage 1 — Wiki page generation.

Per-document pipeline: Writer → Evaluator → Editor (max_rounds configurable).

Failsafe: if the Evaluator fails (timeout, parse error), the draft is
automatically approved so that the pipeline never stalls on a single document.
Atomic write: pages are written via temp+rename to prevent partial files on crash.
Incremental mode: skips documents whose destination page already exists unless
force=True is set.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Template
from markdown_hero import normalize as md_normalize

from ..llm.base import BaseLLMClient
from ..llm.log import LLMLogger
from ..models.config import EntityTypeConfig, WikiConfig
from ..models.document import Document
from ..models.evaluation import PageEvaluation
from ._utils import CHARS_INVALID, write_atomic

logger = logging.getLogger(__name__)


def _safe_filename(s: str) -> str:
    return "".join(c if c not in CHARS_INVALID else "-" for c in s).strip(". ")


def _title_from_draft(draft: str) -> str | None:
    """Extract the first H1 heading from a Markdown draft, or None."""
    for line in draft.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# "):
            return stripped[2:].strip().upper()
    return None


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _render_prompt(prompt_path: Path, context: dict[str, Any]) -> str:
    template_src = prompt_path.read_text(encoding="utf-8")
    return Template(template_src).render(**context)


def _raw_id(doc_id: str) -> str:
    return f"{doc_id}_raw"


def _write_raw_page(doc: Document, entity_cfg: EntityTypeConfig, wiki_dir: Path, generated_at: str) -> Path:
    """Write the original (unstripped) Markdown content to wiki/subdir/raw/<uuid>_raw.md.

    The raw page preserves the source document exactly as converted from its
    original format, before any LLM summarization or cleaning.  It is linked
    from the generated wiki page via a wikilink, providing traceability back
    to the source.  The frontmatter includes content_sha256 so readers can
    verify integrity without re-hashing.

    Args:
        doc: The Document whose content and metadata are written to the raw page.
        entity_cfg: EntityTypeConfig that determines the subdirectory path.
        wiki_dir: Root wiki directory.
        generated_at: ISO-8601 timestamp string to embed in the frontmatter.

    Returns:
        The Path of the written raw page file.
    """
    raw_dir = wiki_dir / entity_cfg.wiki_subdir / "raw"
    raw_path = raw_dir / f"{doc.metadata.id}_raw.md"
    source_name = doc.metadata.extra.get("source_filename") or (doc.content_path.name if doc.content_path else "")
    sha256 = doc.metadata.extra.get("content_sha256", "")
    fm_lines = [
        "---",
        f'id: "{_raw_id(doc.metadata.id)}"',
        f'title: "{doc.metadata.title.replace(chr(34), chr(39))} (original)"',
        f"entity_type: {entity_cfg.slug}",
        "type: raw",
        f'source_file: "{source_name}"',
        f'content_sha256: "{sha256}"',
        f"generated_at: {generated_at}",
        "---",
        "",
        f"# {doc.metadata.title} — Original Content",
        "",
        f"> **Source file:** `{source_name}`",
        f"> **Content UUID:** `{doc.metadata.id}`",
        f"> **SHA-256:** `{sha256}`",
        f"> **Generated at:** {generated_at}",
        "",
        "---",
        "",
    ]
    full = "\n".join(fm_lines) + md_normalize(doc.content) + "\n"
    write_atomic(raw_path, full)
    return raw_path


def _build_frontmatter(doc: Document, entity_cfg: EntityTypeConfig, generated_at: str) -> str:
    """Build the YAML frontmatter block for a generated wiki page.

    Includes the content-addressable UUID, title, entity type, status,
    any domain-specific frontmatter fields from EntityTypeConfig, the source
    filename, a SHA-256 checksum for integrity verification, a wikilink to the
    corresponding raw page, and the generation timestamp.

    Args:
        doc: The source Document whose metadata populates the frontmatter.
        entity_cfg: EntityTypeConfig that specifies which extra fields to emit.
        generated_at: ISO-8601 timestamp string.

    Returns:
        A YAML frontmatter block string terminated with a newline.
    """
    lines = ["---"]
    lines.append(f'id: "{doc.metadata.id}"')
    lines.append(f'title: "{doc.metadata.title.replace(chr(34), chr(39))}"')
    lines.append(f"entity_type: {entity_cfg.slug}")
    lines.append(f"status: {doc.metadata.status}")
    for field in entity_cfg.frontmatter_fields:
        val = doc.metadata.extra.get(field, "")
        lines.append(f"{field}: {val}")
    source_name = doc.metadata.extra.get("source_filename") or (doc.content_path.name if doc.content_path else "")
    sha256 = doc.metadata.extra.get("content_sha256", "")
    lines.append(f'source_file: "{source_name}"')
    lines.append(f'content_sha256: "{sha256}"')
    lines.append(f'source_raw: "[[{_raw_id(doc.metadata.id)}]]"')
    lines.append(f"generated_at: {generated_at}")
    lines.append("---")
    return "\n".join(lines) + "\n"


async def _generate_draft(
    doc: Document,
    entity_cfg: EntityTypeConfig,
    llm: BaseLLMClient,
    cfg: WikiConfig,
    llm_logger: LLMLogger,
) -> str:
    """Call the Writer LLM to produce an initial wiki page draft.

    Renders the entity type's generate prompt with the document content and
    metadata, then sends it to the LLM.  Input is truncated to cfg.max_chars_input
    if the document exceeds the context window limit.

    Args:
        doc: The source Document.
        entity_cfg: Provides the generate prompt path.
        llm: Active LLM client.
        cfg: Pipeline config for max_chars_input and model settings.
        llm_logger: Logger to record this call.

    Returns:
        The raw LLM response text (the initial draft).
    """
    content, truncated = _truncate(md_normalize(doc.content), cfg.max_chars_input)
    if truncated:
        logger.warning("[TRUNCATED] %s: %d → %d chars", doc.metadata.id, len(doc.content), cfg.max_chars_input)

    context = {
        "document": content,
        "metadata": doc.metadata.model_dump(),
        "entity_type": entity_cfg.name,
        "language": cfg.language,
    }
    system = _render_prompt(entity_cfg.prompt_generate, context)
    user = content

    t0 = llm_logger.start_call()
    resp = await llm.call(system, user)
    llm_logger.record(
        system=system, user=user, output=resp.text,
        tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
        cached_tokens=resp.cached_tokens, model_id=resp.model_id,
        stage="generate.writer", elapsed=time.monotonic() - t0,
    )
    return resp.text


async def _evaluate_draft(
    draft: str,
    doc: Document,
    entity_cfg: EntityTypeConfig,
    llm: BaseLLMClient,
    cfg: WikiConfig,
    llm_logger: LLMLogger,
) -> PageEvaluation:
    """Call the Evaluator LLM to assess draft quality and return structured feedback.

    Uses call_structured with PageEvaluation as the output schema so that the
    LLM response is validated by Pydantic before being used.  If parsing fails
    (timeout, malformed JSON, schema violation), returns PageEvaluation(approved=True)
    as a failsafe so that the pipeline always produces output.

    Args:
        draft: The current wiki page draft text.
        doc: The source Document (metadata used in the prompt context).
        entity_cfg: Provides the evaluate prompt path.
        llm: Active LLM client.
        cfg: Pipeline config for model_id (used in error logging).
        llm_logger: Logger to record this call.

    Returns:
        A PageEvaluation instance with approved, problems, and suggestions.
    """
    context = {
        "draft": draft,
        "metadata": doc.metadata.model_dump(),
        "entity_type": entity_cfg.name,
        "language": cfg.language,
    }
    system = _render_prompt(entity_cfg.prompt_evaluate, context)
    user = draft

    t0 = llm_logger.start_call()
    try:
        result = await llm.call_structured(system, user, PageEvaluation)
        resp_text = result.model_dump_json()
        llm_logger.record(
            system=system, user=user, output=resp_text,
            tokens_in=None, tokens_out=None, cached_tokens=None,
            model_id=cfg.llm.model_id, stage="generate.evaluator",
            elapsed=time.monotonic() - t0,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evaluator failed for %s (%s) — auto-approving", doc.metadata.id, exc)
        llm_logger.record(
            system=system, user=user, output="", tokens_in=None, tokens_out=None,
            cached_tokens=None, model_id=cfg.llm.model_id,
            stage="generate.evaluator", status="error", error=str(exc),
            elapsed=time.monotonic() - t0,
        )
        return PageEvaluation(approved=True)


async def _edit_draft(
    draft: str,
    evaluation: PageEvaluation,
    doc: Document,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
) -> str:
    problems = "\n".join(f"- {p}" for p in evaluation.problems)
    suggestions = "\n".join(f"- {s}" for s in evaluation.suggestions)
    context = {
        "draft": draft,
        "problems": problems,
        "suggestions": suggestions,
        "language": cfg.language,
    }
    system = _render_prompt(cfg.prompt_editor, context)
    user = f"**Problems:**\n{problems}\n\n**Suggestions:**\n{suggestions}\n\n**Draft:**\n{draft}"

    t0 = llm_logger.start_call()
    resp = await llm.call(system, user)
    llm_logger.record(
        system=system, user=user, output=resp.text,
        tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
        cached_tokens=resp.cached_tokens, model_id=resp.model_id,
        stage="generate.editor", elapsed=time.monotonic() - t0,
    )
    return resp.text


async def generate_page(
    doc: Document,
    cfg: WikiConfig,
    llm: BaseLLMClient,
    llm_logger: LLMLogger,
    *,
    force: bool = False,
) -> Path | None:
    """Generate (or regenerate) the wiki page for a single document.

    Orchestrates the Writer → Evaluator → Editor loop, writes the raw page,
    assembles the final page with frontmatter and a source section, then
    writes it atomically.  In incremental mode (force=False), skips the
    document if the destination file already exists.

    Args:
        doc: The source Document to generate a page for.
        cfg: Active WikiConfig with wiki_dir, prompts, and error handling settings.
        llm: Active LLM client.
        llm_logger: Logger for all LLM calls made during generation.
        force: When True, regenerate the page even if it already exists.

    Returns:
        The Path of the generated .md file on success, or None if generation
        failed and on_llm_error is set to ``"skip"``.

    Raises:
        Exception: Any LLM or I/O error when on_llm_error is set to ``"abort"``.
    """
    entity_cfg = cfg.entity_type_by_slug(doc.metadata.entity_type)
    if entity_cfg is None:
        entity_cfg = cfg.entity_types[0]
        logger.warning(
            "entity_type '%s' unknown for %s — using '%s'",
            doc.metadata.entity_type, doc.metadata.id, entity_cfg.slug,
        )

    subdir = cfg.wiki_dir / entity_cfg.wiki_subdir

    # Incremental pre-check: use the source metadata title as a best-effort guess
    # for the final filename so we can skip the LLM call when the page already exists.
    _preliminary_slug = _safe_filename(doc.metadata.title or doc.metadata.id)
    _preliminary_dest = subdir / f"{_preliminary_slug}.md"
    if _preliminary_dest.exists() and not force:
        logger.debug("Incremental: skipping %s (already exists)", _preliminary_dest.name)
        return _preliminary_dest

    generated_at = datetime.now(UTC).isoformat()

    try:
        draft = await _generate_draft(doc, entity_cfg, llm, cfg, llm_logger)

        for round_n in range(1, entity_cfg.max_rounds + 1):
            evaluation = await _evaluate_draft(draft, doc, entity_cfg, llm, cfg, llm_logger)
            if evaluation.approved:
                break
            if round_n < entity_cfg.max_rounds:
                draft = await _edit_draft(draft, evaluation, doc, cfg, llm, llm_logger)

        # Derive filename from the H1 title the LLM wrote; fallback to UUID
        page_title = _title_from_draft(draft)
        if page_title:
            dest = subdir / f"{_safe_filename(page_title)}.md"
        else:
            dest = subdir / f"{_safe_filename(doc.metadata.id)}.md"
            logger.warning("No H1 found in draft for %s — using UUID filename", doc.metadata.id)

        if dest.exists() and not force:
            logger.debug("Incremental: skipping %s (already exists)", dest.name)
            return dest

        # Write original content before the wiki page
        _write_raw_page(doc, entity_cfg, cfg.wiki_dir, generated_at)

        raw_link = f"[Original Content](raw/{_raw_id(doc.metadata.id)}.md)"
        source_section = (
            "\n\n---\n\n"
            "## Original Document\n\n"
            f"> Full content of the source document: {raw_link}\n"
        )

        frontmatter = _build_frontmatter(doc, entity_cfg, generated_at)
        full_page = frontmatter + "\n" + draft.strip() + source_section
        write_atomic(dest, full_page)
        logger.info("Generated: %s", dest)
        return dest

    except Exception as exc:  # noqa: BLE001
        if cfg.on_llm_error == "abort":
            raise
        logger.error("Error generating %s: %s — skipping", doc.metadata.id, exc)
        return None
