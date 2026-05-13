"""Async orchestrator for the wiki-llm pipeline.

Executes up to eight stages in order:
  1. read        — discover and convert documents from content_dir
  2. generate    — Writer → Evaluator → Editor loop per document
  3. taxonomy    — collect terms + generate taxonomy pages
  4. groups      — generate organizational grouping pages
  5. index       — rebuild index.md
  6. consolidate — semantic merge of duplicate pages
  7. lint        — static + semantic quality check
  8. repair      — LangGraph repair agent for broken links / orphans
  9. export      — (optional) export all pages to .docx via markdown_hero

Each stage is opt-in via PipelineOptions.stages so that individual CLI
commands can run only the stages they need.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .llm.factory import create_client
from .llm.log import LLMLogger
from .models.config import WikiConfig
from .models.document import Document
from .readers.filesystem import FilesystemReader
from .stages.generate import generate_page
from .stages.taxonomy import run_taxonomy
from .stages.groups import run_groups
from .stages.index import run_index
from .stages.consolidate import run_consolidate
from .stages.lint import run_lint
from .stages.repair import run_repair

logger = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    """Runtime options that control pipeline behaviour for a single run.

    Attributes:
        force: When True, regenerate wiki pages that already exist on disk.
        workers: Maximum number of concurrent generate tasks (asyncio Semaphore).
        stages: Ordered list of stage names to execute.  Stages not in this
            list are skipped entirely.
    """

    force: bool = False
    workers: int = 4
    stages: list[str] = field(default_factory=lambda: [
        "read", "generate", "taxonomy", "groups", "index",
        "consolidate", "lint", "repair",
    ])


@dataclass
class PipelineResult:
    """Counters and timing collected by run_pipeline for display / testing.

    Attributes:
        docs_read: Total documents discovered by the reader.
        pages_generated: Documents that produced a wiki page successfully.
        pages_skipped: Documents skipped because their page already existed
            and force=False.
        pages_error: Documents that failed during the generate stage.
        elapsed_s: Wall-clock seconds from pipeline start to finish.
    """

    docs_read: int = 0
    pages_generated: int = 0
    pages_skipped: int = 0
    pages_error: int = 0
    elapsed_s: float = 0.0


async def run_pipeline(cfg: WikiConfig, opts: PipelineOptions | None = None) -> PipelineResult:
    """Execute the wiki-llm pipeline and return an aggregated result summary.

    Creates a single LLM client and LLMLogger shared across all stages, then
    runs each enabled stage in sequence.  The generate stage uses
    asyncio.gather with a Semaphore to bound concurrency.  After generate,
    source files are moved to content_processed_dir on success or
    content_error_dir on failure.

    Args:
        cfg: Active WikiConfig describing content dirs, wiki dir, LLM settings,
            entity types, taxonomies, groupings, and prompt paths.
        opts: Runtime options (force, workers, stages).  Defaults to a
            PipelineOptions with all stages enabled and 4 workers.

    Returns:
        A PipelineResult with document counts and total elapsed time.
    """
    if opts is None:
        opts = PipelineOptions()

    stages = set(opts.stages)
    result = PipelineResult()
    t_start = time.monotonic()

    llm_logger = LLMLogger(cfg.log_dir)
    llm = create_client(cfg.llm)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=False,
    )

    with progress:

        # ------------------------------------------------------------------
        # 1. Read
        # ------------------------------------------------------------------
        docs: list[Document] = []
        if "read" in stages:
            t_read = progress.add_task("[read] scanning documents…", total=None)
            reader = FilesystemReader(cfg)
            docs = await reader.read_all()
            result.docs_read = len(docs)

            if cfg.status_filter:
                docs = [d for d in docs if d.metadata.status in cfg.status_filter or not d.metadata.status]

            progress.update(t_read, description=f"[read] {result.docs_read} documents", total=1, completed=1)

        # ------------------------------------------------------------------
        # 2. Generate
        # ------------------------------------------------------------------
        if "generate" in stages and docs:
            semaphore = asyncio.Semaphore(opts.workers)
            cfg.wiki_dir.mkdir(parents=True, exist_ok=True)
            processed_dir = cfg.get_processed_dir()
            error_dir = cfg.get_error_dir()

            t_gen = progress.add_task(
                f"[generate] 0/{len(docs)} pages",
                total=len(docs),
                completed=0,
            )

            async def _gen_one(doc: Document) -> tuple[Document, Path | None]:
                async with semaphore:
                    try:
                        page = await generate_page(doc, cfg, llm, llm_logger, force=opts.force)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("generate exception for %s: %s", doc.metadata.id, exc)
                        page = None
                    finally:
                        done = int(progress.tasks[t_gen].completed) + 1
                        progress.update(
                            t_gen,
                            advance=1,
                            description=f"[generate] {done}/{len(docs)} pages",
                        )
                    return doc, page

            tasks = [_gen_one(doc) for doc in docs]
            pair_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in pair_results:
                if isinstance(r, Exception):
                    result.pages_error += 1
                    logger.debug("generate unexpected exception: %s", r)
                    continue
                doc, page = r
                if page is None:
                    result.pages_error += 1
                    if doc.content_path and doc.content_path.exists():
                        _move_source(doc.content_path, error_dir)
                else:
                    result.pages_generated += 1
                    if doc.content_path and doc.content_path.exists():
                        _move_source(doc.content_path, processed_dir)

        # ------------------------------------------------------------------
        # 3. Taxonomy
        # ------------------------------------------------------------------
        if "taxonomy" in stages and cfg.taxonomies:
            t_tax = progress.add_task("[taxonomy] normalizing terms…", total=None)
            await run_taxonomy(cfg, llm, llm_logger)
            progress.update(t_tax, description="[taxonomy] done", total=1, completed=1)

        # ------------------------------------------------------------------
        # 4. Groups
        # ------------------------------------------------------------------
        if "groups" in stages and cfg.groupings:
            t_grp = progress.add_task("[groups] building grouping pages…", total=None)
            await run_groups(cfg, docs)
            progress.update(t_grp, description="[groups] done", total=1, completed=1)

        # ------------------------------------------------------------------
        # 5. Index
        # ------------------------------------------------------------------
        if "index" in stages:
            t_idx = progress.add_task("[index] rebuilding index…", total=None)
            await run_index(cfg)
            progress.update(t_idx, description="[index] done", total=1, completed=1)

        # ------------------------------------------------------------------
        # 6. Consolidate
        # ------------------------------------------------------------------
        if "consolidate" in stages:
            t_con = progress.add_task("[consolidate] merging duplicates…", total=None)
            await run_consolidate(cfg, llm, llm_logger)
            progress.update(t_con, description="[consolidate] done", total=1, completed=1)

        # ------------------------------------------------------------------
        # 7. Lint
        # ------------------------------------------------------------------
        repair_state = None
        if "lint" in stages:
            t_lint = progress.add_task("[lint] checking quality…", total=None)
            repair_state = await run_lint(cfg, llm, llm_logger)
            progress.update(t_lint, description="[lint] done", total=1, completed=1)

        # ------------------------------------------------------------------
        # 8. Repair
        # ------------------------------------------------------------------
        if "repair" in stages and repair_state is not None:
            t_rep = progress.add_task("[repair] fixing broken links…", total=None)
            repair_state = await run_repair(repair_state, cfg, llm, llm_logger)
            progress.update(t_rep, description="[repair] done", total=1, completed=1)

        # ------------------------------------------------------------------
        # 9. Export (Word)
        # ------------------------------------------------------------------
        if cfg.export_word:
            t_exp = progress.add_task("[export] generating .docx files…", total=None)
            _export_word(cfg)
            progress.update(t_exp, description="[export] done", total=1, completed=1)

    result.elapsed_s = time.monotonic() - t_start
    logger.debug(
        "Pipeline complete in %.1fs | read=%d generated=%d errors=%d",
        result.elapsed_s, result.docs_read, result.pages_generated, result.pages_error,
    )
    return result


def _export_word(cfg: WikiConfig) -> None:
    """Export every wiki Markdown file to a .docx file via markdown_hero.word_format.

    Writes output files to wiki_dir/export_word/.  Skips files whose .docx
    already exists (incremental).  Silently logs a warning and continues if
    word_format fails for an individual page so that one bad page cannot abort
    the full export.

    Args:
        cfg: Active WikiConfig; uses wiki_dir as the source tree.
    """
    try:
        from markdown_hero import word_format  # noqa: PLC0415
        out_dir = cfg.wiki_dir / "export_word"
        out_dir.mkdir(parents=True, exist_ok=True)
        for md_file in cfg.wiki_dir.rglob("*.md"):
            docx_path = out_dir / (md_file.stem + ".docx")
            if not docx_path.exists():
                try:
                    word_format(md_file.read_text(encoding="utf-8"), str(docx_path))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("word_format failed for %s: %s", md_file.name, exc)
        logger.info("Word export complete: %s", out_dir)
    except ImportError:
        logger.warning("markdown_hero.word_format not available — export_word skipped")


def _move_source(source: Path, dest_dir: Path) -> None:
    """Move a source file to dest_dir after processing, handling name collisions.

    If a file with the same name already exists in dest_dir, appends a 6-char
    SHA-1 suffix derived from the source path to avoid silent overwrites.  All
    exceptions are caught and logged as warnings so that a failed move never
    interrupts the pipeline.

    Args:
        source: Path of the file to move.
        dest_dir: Destination directory (created if it does not exist).
    """
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        if dest.exists():
            h = hashlib.sha1(str(source).encode()).hexdigest()[:6]
            dest = dest_dir / f"{source.stem}_{h}{source.suffix}"
        shutil.move(str(source), dest)
        logger.debug("Moved: %s → %s", source.name, dest_dir.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not move %s: %s", source, exc)
