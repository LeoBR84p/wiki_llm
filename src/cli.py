"""wiki-llm pipeline CLI — powered by Typer.

Available commands:
  generate      read + generate wiki pages
  topics        taxonomy collection and normalization
  groups        organizational grouping pages
  index         rebuild index.md
  consolidate   semantic merge + LLM deduplication
  lint          static + semantic lint analysis
  repair        lint + LangGraph repair agent
  run-all       all stages in order
  chat          start NiceGUI RAG chat server

Global options:
  --config PATH       Path to a Python WikiConfig file
  --force             Regenerate already-existing pages
  --workers N         Parallelism level for page generation
  --no-interactive    Skip confirmation prompts

Config convention:
  The Python config file must expose an object named ``config``
  (a WikiConfig instance) or a callable ``get_config() -> WikiConfig``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .models.config import WikiConfig
from .pipeline import PipelineOptions, run_pipeline

app = typer.Typer(
    name="wiki-llm",
    help="LLM-powered wiki generation pipeline.",
    add_completion=False,
)

_DEFAULT_CONFIG = Path("config/wiki_config.py")


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logging for the CLI session.

    Sets DEBUG level when verbose is True, WARNING otherwise (errors only).
    Uses RichHandler for clean output that coexists with the progress bar.
    Third-party chatty loggers (openai, httpx) are silenced unless verbose.

    Args:
        verbose: When True, enable DEBUG-level log output.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Always silence very chatty third-party loggers
    for noisy in ("openai", "openai._base_client", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    # src loggers: DEBUG in verbose mode, WARNING otherwise
    logging.getLogger("src").setLevel(logging.DEBUG if verbose else logging.WARNING)


def _load_config(config_path: Path) -> WikiConfig:
    """Load a WikiConfig from a Python file at the given path.

    The file must expose either a ``config`` attribute (WikiConfig instance)
    or a ``get_config()`` callable that returns a WikiConfig.  This design
    lets users write plain Python — no YAML/TOML parsing, full IDE support.

    Args:
        config_path: Absolute or relative path to the Python config module.

    Returns:
        A validated WikiConfig instance.

    Raises:
        typer.Exit: If the file does not exist, cannot be imported, or does
            not expose the expected interface.
    """
    if not config_path.exists():
        typer.echo(f"[ERROR] Config file not found: {config_path}", err=True)
        raise typer.Exit(1)

    spec = importlib.util.spec_from_file_location("_wiki_config", config_path)
    if spec is None or spec.loader is None:
        typer.echo(f"[ERROR] Could not load: {config_path}", err=True)
        raise typer.Exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "config"):
        return module.config
    if hasattr(module, "get_config") and callable(module.get_config):
        return module.get_config()

    typer.echo(
        f"[ERROR] {config_path} must expose `config: WikiConfig` or `get_config() -> WikiConfig`",
        err=True,
    )
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Shared options via callback
# ---------------------------------------------------------------------------

_state: dict = {}


@app.callback()
def main_callback(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c", help="Python WikiConfig file"),
    force: bool = typer.Option(False, "--force", help="Regenerate existing pages"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers for page generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    no_interactive: bool = typer.Option(False, "--no-interactive", help="Skip confirmation prompts"),
) -> None:
    """Store global CLI options in module-level state for sub-commands to read.

    Typer calls this callback before any sub-command.  Storing options in
    ``_state`` avoids threading them through every function signature.

    Args:
        config: Path to the Python WikiConfig file.
        force: Whether to regenerate pages that already exist on disk.
        workers: Number of asyncio tasks to run concurrently during generate.
        verbose: Enable DEBUG logging.
        no_interactive: Suppress any interactive confirmation prompts.
    """
    _setup_logging(verbose)
    _state["config_path"] = config
    _state["force"] = force
    _state["workers"] = workers
    _state["no_interactive"] = no_interactive


def _get_opts(stages: list[str]) -> tuple[WikiConfig, PipelineOptions]:
    """Build a (WikiConfig, PipelineOptions) pair from current CLI state.

    Reads the config file registered by the callback and wraps the global
    options into a PipelineOptions dataclass.  Called at the start of every
    sub-command so that each command gets a fresh, validated config.

    Args:
        stages: List of pipeline stage names to include in this run.

    Returns:
        A tuple of (WikiConfig, PipelineOptions) ready to pass to run_pipeline.
    """
    cfg = _load_config(_state.get("config_path", _DEFAULT_CONFIG))
    opts = PipelineOptions(
        force=_state.get("force", False),
        workers=_state.get("workers", 4),
        stages=stages,
    )
    return cfg, opts


def _run(stages: list[str]) -> None:
    """Execute the pipeline for the given stages and print a summary line.

    Delegates to ``run_pipeline`` via asyncio.run, then echoes elapsed time
    and document counts so the user has immediate feedback in the terminal.

    Args:
        stages: Ordered list of stage names to execute (e.g. ["read", "generate"]).
    """
    cfg, opts = _get_opts(stages)
    result = asyncio.run(run_pipeline(cfg, opts))
    typer.echo(
        f"Done in {result.elapsed_s:.1f}s | "
        f"read={result.docs_read} generated={result.pages_generated} errors={result.pages_error}"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def generate(
    ctx: typer.Context,
) -> None:
    """Read documents from content_dir and generate wiki pages.

    Runs the ``read`` and ``generate`` stages.  Each source file is converted
    to Markdown, assigned a content-addressable UUID, and then passed through
    the Writer -> Evaluator -> Editor loop.  Successfully processed files are
    moved to content_processed_dir; failed ones to content_error_dir.
    """
    _run(["read", "generate"])


@app.command()
def topics() -> None:
    """Collect terms from wiki pages, normalize via LLM, and generate taxonomy pages.

    Scans every wiki page for terms listed under the configured section header,
    deduplicates and normalizes them in batches using the LLM, then writes one
    Markdown page per canonical term into the taxonomy subdirectory.
    """
    _run(["taxonomy"])


@app.command()
def groups() -> None:
    """Generate organizational grouping pages from document metadata.

    For each GroupingConfig, reads the configured metadata field from every
    Document and produces one summary page per distinct value in the grouping
    wiki subdirectory (e.g. one page per team or department).
    """
    cfg, _ = _get_opts([])
    from .stages.groups import run_groups  # noqa: PLC0415
    asyncio.run(run_groups(cfg, []))


@app.command()
def index() -> None:
    """Rebuild the global wiki index.md.

    Scans all wiki subdirectories and writes a fresh index.md with sections
    for entity types, taxonomies, and groupings.  Safe to run multiple times
    (idempotent overwrite).
    """
    cfg, _ = _get_opts([])
    from .stages.index import run_index  # noqa: PLC0415
    asyncio.run(run_index(cfg))
    typer.echo("index.md generated.")


@app.command()
def consolidate() -> None:
    """Run semantic consolidation: markdown_merge pre-pass + LLM deduplication.

    First merges structurally similar pages using markdown_hero.markdown_merge,
    then asks the LLM to identify semantic duplicates in batches of up to 80
    pages.  Duplicate pages are renamed to the canonical title and all
    wikilinks across the wiki are updated accordingly.
    """
    _run(["consolidate"])


@app.command()
def lint() -> None:
    """Run static lint (markdown_hero) + semantic LLM analysis on all wiki pages.

    Detects orphan pages (no inbound links), broken wikilinks, skipped heading
    levels, and unclosed fences via markdown_hero.lint().  Optionally calls the
    LLM to evaluate content quality.  Writes a full report to lint_report.md.
    """
    _run(["lint"])


@app.command()
def repair() -> None:
    """Run lint and then the LangGraph repair agent.

    After the static lint pass, dispatches each broken link and orphan page
    to a LangGraph agent (Send fan-out pattern) that either creates stub pages
    for broken links or adds backlinks for orphaned pages.
    """
    _run(["lint", "repair"])


@app.command(name="run-all")
def run_all(
    ctx: typer.Context,
) -> None:
    """Run every pipeline stage in order.

    Executes: read -> generate -> taxonomy -> groups -> index ->
    consolidate -> lint -> repair.  Equivalent to chaining all individual
    commands in a single call, sharing the same LLM client and logger.
    """
    _run(["read", "generate", "taxonomy", "groups", "index", "consolidate", "lint", "repair"])


@app.command()
def chat(
    host: str = typer.Option("0.0.0.0", "--host", help="Chat server host address"),
    port: int = typer.Option(8080, "--port", help="Chat server port"),
) -> None:
    """Start the NiceGUI RAG chat server.

    Builds an in-memory BM25 index over all wiki pages using
    markdown_hero.extract_chunks, then serves a NiceGUI chat interface that
    retrieves relevant chunks and calls the LLM with full conversation history.
    Override host/port via --host and --port or the WIKI_UI_HOST / WIKI_UI_PORT
    environment variables.

    Args:
        host: Network interface to bind to (default 0.0.0.0 for Docker/K8s).
        port: TCP port for the NiceGUI HTTP server.
    """
    try:
        from .ui.app import start_ui  # noqa: PLC0415
    except ImportError as exc:
        typer.echo(f"[ERROR] UI not available: {exc}", err=True)
        raise typer.Exit(1) from exc

    cfg, _ = _get_opts([])
    start_ui(cfg, host=host, port=port)
