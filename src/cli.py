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
    """Load WikiConfig from a Python file exposing `config` or `get_config()`."""
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
    typer.echo(
        "\n"
        "██╗    ██╗██╗██╗  ██╗██╗    ██╗     ██╗     ███╗   ███╗\n"
        "██║    ██║██║██║ ██╔╝██║    ██║     ██║     ████╗ ████║\n"
        "██║ █╗ ██║██║█████╔╝ ██║    ██║     ██║     ██╔████╔██║\n"
        "██║███╗██║██║██╔═██╗ ██║    ██║     ██║     ██║╚██╔╝██║\n"
        "╚███╔███╔╝██║██║  ██╗██║    ███████╗███████╗██║ ╚═╝ ██║\n"
        " ╚══╝╚══╝ ╚═╝╚═╝  ╚═╝╚═╝    ╚══════╝╚══════╝╚═╝     ╚═╝\n"
        "                                                   by leobr.site\n"
    )
    _setup_logging(verbose)
    _state["config_path"] = config
    _state["force"] = force
    _state["workers"] = workers
    _state["no_interactive"] = no_interactive


def _get_opts(stages: list[str]) -> tuple[WikiConfig, PipelineOptions]:
    cfg = _load_config(_state.get("config_path", _DEFAULT_CONFIG))
    opts = PipelineOptions(
        force=_state.get("force", False),
        workers=_state.get("workers", 4),
        stages=stages,
    )
    return cfg, opts


def _run(stages: list[str]) -> None:
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
def generate(ctx: typer.Context) -> None:
    """Read documents from content_dir and generate wiki pages."""
    _run(["read", "generate"])


@app.command()
def topics() -> None:
    """Collect terms from wiki pages, normalize via LLM, and generate taxonomy pages."""
    _run(["taxonomy"])


@app.command()
def groups() -> None:
    """Generate organizational grouping pages from document metadata."""
    cfg, _ = _get_opts([])
    from .stages.groups import run_groups  # noqa: PLC0415
    asyncio.run(run_groups(cfg, []))


@app.command()
def index() -> None:
    """Rebuild the global wiki index.md."""
    cfg, _ = _get_opts([])
    from .stages.index import run_index  # noqa: PLC0415
    asyncio.run(run_index(cfg))
    typer.echo("index.md generated.")


@app.command()
def consolidate() -> None:
    """Semantic merge: structural pre-pass + LLM deduplication."""
    _run(["consolidate"])


@app.command()
def lint() -> None:
    """Static + semantic LLM analysis; writes lint_report.md."""
    _run(["lint"])


@app.command()
def repair() -> None:
    """Run lint then the LangGraph repair agent (broken links + orphans)."""
    _run(["lint", "repair"])


@app.command(name="run-all")
def run_all(ctx: typer.Context) -> None:
    """Run all pipeline stages in order."""
    _run(["read", "generate", "taxonomy", "groups", "index", "consolidate", "lint", "repair"])


@app.command()
def chat(
    host: str = typer.Option("0.0.0.0", "--host", help="Chat server host address"),
    port: int = typer.Option(8080, "--port", help="Chat server port"),
) -> None:
    """Start the NiceGUI RAG chat server."""
    try:
        from .ui.app import start_ui  # noqa: PLC0415
    except ImportError as exc:
        typer.echo(f"[ERROR] UI not available: {exc}", err=True)
        raise typer.Exit(1) from exc

    cfg, _ = _get_opts([])
    start_ui(cfg, host=host, port=port)


@app.command()
def setup() -> None:
    """Run the interactive setup wizard to generate config/my_wiki.py and .env."""
    from .setup_wizard import run_wizard  # noqa: PLC0415
    run_wizard()
