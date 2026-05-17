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
from ..models.config import ObjectTypeConfig, WikiConfig
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


def _write_raw_page(doc: Document, obj_cfg: ObjectTypeConfig, wiki_dir: Path, generated_at: str) -> Path:
    """Write the original (unstripped) Markdown content to wiki/subdir/raw/<uuid>_raw.md.

    Args:
        doc: The Document whose content and metadata are written to the raw page.
        obj_cfg: ObjectTypeConfig that determines the subdirectory path.
        wiki_dir: Root wiki directory.
        generated_at: ISO-8601 timestamp string to embed in the frontmatter.

    Returns:
        The Path of the written raw page file.
    """
    raw_dir = wiki_dir / obj_cfg.wiki_subdir / "raw"
    raw_path = raw_dir / f"{doc.metadata.id}_raw.md"
    source_name = doc.metadata.extra.get("source_filename") or (doc.content_path.name if doc.content_path else "")
    sha256 = doc.metadata.extra.get("content_sha256", "")
    fm_lines = [
        "---",
        f'id: "{_raw_id(doc.metadata.id)}"',
        f'title: "{doc.metadata.title.replace(chr(34), chr(39))} (original)"',
        f"object_type: {obj_cfg.slug}",
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


def _build_frontmatter(doc: Document, obj_cfg: ObjectTypeConfig, generated_at: str) -> str:
    """Build the YAML frontmatter block for a generated wiki page.

    Args:
        doc: The source Document whose metadata populates the frontmatter.
        obj_cfg: ObjectTypeConfig that specifies which extra fields to emit.
        generated_at: ISO-8601 timestamp string.

    Returns:
        A YAML frontmatter block string terminated with a newline.
    """
    lines = ["---"]
    lines.append(f'id: "{doc.metadata.id}"')
    lines.append(f'title: "{doc.metadata.title.replace(chr(34), chr(39))}"')
    lines.append(f"object_type: {obj_cfg.slug}")
    lines.append(f"status: {doc.metadata.status}")
    for field in obj_cfg.frontmatter_fields:
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
    obj_cfg: ObjectTypeConfig,
    llm: BaseLLMClient,
    cfg: WikiConfig,
    llm_logger: LLMLogger,
) -> str:
    """Call the Writer LLM to produce an initial wiki page draft.

    Args:
        doc: The source Document.
        obj_cfg: Provides the generate prompt path.
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
        "object_type": obj_cfg.name,
        "language": cfg.language,
    }
    system = _render_prompt(obj_cfg.prompt_generate, context)
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
    obj_cfg: ObjectTypeConfig,
    llm: BaseLLMClient,
    cfg: WikiConfig,
    llm_logger: LLMLogger,
) -> PageEvaluation:
    """Call the Evaluator LLM to assess draft quality and return structured feedback.

    Args:
        draft: The current wiki page draft text.
        doc: The source Document (metadata used in the prompt context).
        obj_cfg: Provides the evaluate prompt path.
        llm: Active LLM client.
        cfg: Pipeline config for model_id (used in error logging).
        llm_logger: Logger to record this call.

    Returns:
        A PageEvaluation instance with approved, problems, and suggestions.
    """
    context = {
        "draft": draft,
        "metadata": doc.metadata.model_dump(),
        "object_type": obj_cfg.name,
        "language": cfg.language,
    }
    system = _render_prompt(obj_cfg.prompt_evaluate, context)
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
        "problems": problems,
        "suggestions": suggestions,
        "language": cfg.language,
    }
    system = _render_prompt(cfg.prompt_editor, context)
    user = draft

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
    obj_cfg = cfg.object_by_slug(doc.metadata.object_type)
    if obj_cfg is None:
        obj_cfg = cfg.objects[0]
        logger.warning(
            "object_type '%s' unknown for %s — using '%s'",
            doc.metadata.object_type, doc.metadata.id, obj_cfg.slug,
        )

    subdir = cfg.wiki_dir / obj_cfg.wiki_subdir

    generated_at = datetime.now(UTC).isoformat()

    # Write raw page (1-to-1 with every successfully read document),
    # regardless of LLM outcome or incremental skip.
    raw_path = cfg.wiki_dir / obj_cfg.wiki_subdir / "raw" / f"{doc.metadata.id}_raw.md"
    if not raw_path.exists() or force:
        _write_raw_page(doc, obj_cfg, cfg.wiki_dir, generated_at)

    if not force and subdir.exists():
        existing = next(
            (p for p in subdir.glob("*.md")
             if f'id: "{doc.metadata.id}"' in p.read_text(encoding="utf-8")[:200]),
            None,
        )
        if existing:
            logger.debug("Incremental: skipping %s (id already generated)", doc.metadata.id)
            return existing

    try:
        draft = await _generate_draft(doc, obj_cfg, llm, cfg, llm_logger)

        for round_n in range(1, obj_cfg.max_rounds + 1):
            evaluation = await _evaluate_draft(draft, doc, obj_cfg, llm, cfg, llm_logger)
            if evaluation.approved:
                break
            if round_n < obj_cfg.max_rounds:
                draft = await _edit_draft(draft, evaluation, doc, cfg, llm, llm_logger)

        page_title = _title_from_draft(draft)
        if page_title:
            dest = subdir / f"{_safe_filename(page_title)}.md"
        else:
            dest = subdir / f"{_safe_filename(doc.metadata.id)}.md"
            logger.warning("No H1 found in draft for %s — using UUID filename", doc.metadata.id)

        if dest.exists() and not force:
            logger.debug("Incremental: skipping %s (already exists)", dest.name)
            return dest

        raw_link = f"[Original Content](raw/{_raw_id(doc.metadata.id)}.md)"
        source_section = (
            "\n\n---\n\n"
            "## Original Document\n\n"
            f"> Full content of the source document: {raw_link}\n"
        )

        frontmatter = _build_frontmatter(doc, obj_cfg, generated_at)
        full_page = frontmatter + "\n" + draft.strip() + source_section
        write_atomic(dest, full_page)
        logger.info("Generated: %s", dest)
        return dest

    except Exception as exc:  # noqa: BLE001
        if cfg.on_llm_error == "abort":
            raise
        logger.error("Error generating %s: %s — skipping", doc.metadata.id, exc)
        return None
