"""Interactive setup wizard for wiki-llm.

Invoked via:
    uv run wiki-llm setup

Guides the user through configuring WikiConfig, entity types, taxonomies,
groupings, and generates config/my_wiki.py + a .env template.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import questionary
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOGO = """\
██╗    ██╗██╗██╗  ██╗██╗    ██╗     ██╗     ███╗   ███╗
██║    ██║██║██║ ██╔╝██║    ██║     ██║     ████╗ ████║
██║ █╗ ██║██║█████╔╝ ██║    ██║     ██║     ██╔████╔██║
██║███╗██║██║██╔═██╗ ██║    ██║     ██║     ██║╚██╔╝██║
╚███╔███╔╝██║██║  ██╗██║    ███████╗███████╗██║ ╚═╝ ██║
 ╚══╝╚══╝ ╚═╝╚═╝  ╚═╝╚═╝    ╚══════╝╚══════╝╚═╝     ╚═╝
                                                   by leobr.site"""

_ROOT = Path(__file__).parent.parent
_PROMPTS_DIR = _ROOT / "config" / "prompts"
_CONFIG_PATH = _ROOT / "config" / "my_wiki.py"
_ENV_PATH = _ROOT / ".env"

_BACKEND_KEYS: dict[str, str | None] = {
    "openrouter": "OPENROUTER_APIKEY",
    "openai": "OPENAI_API_KEY",
    "bedrock": "AWS_LOGINKEY",
    "ollama": None,
}

_MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "openrouter": [
        "x-ai/grok-3-mini-beta",
        "google/gemma-4-26b-a4b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "meta-llama/llama-4-scout:free",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
    ],
    "bedrock": [
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "amazon.nova-lite-v1:0",
    ],
    "ollama": [
        "llama3.2",
        "mistral",
        "phi4",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_float(v: str, lo: float = 0.0, hi: float = 1.0) -> bool | str:
    try:
        f = float(v)
        return True if lo <= f <= hi else f"Must be between {lo} and {hi}"
    except ValueError:
        return f"Must be a number between {lo} and {hi}"


def _env_key_present(backend: str) -> bool:
    """Return True if the backend's API key is already set in the environment."""
    key = _BACKEND_KEYS.get(backend)
    if key is None:
        return True  # ollama needs no API key
    load_dotenv(_ENV_PATH, override=False)
    return bool(os.environ.get(key))


def _warn_no_env(backend: str, console: Console) -> bool:
    """Warn about a missing API key and ask whether to skip LLM generation.

    Returns True if the user chooses to skip (use example templates instead).
    """
    key = _BACKEND_KEYS[backend]
    console.print(
        f"\n[bold yellow]  ⚠  LLM step requires [white]{key}[/white] in your .env file.[/bold yellow]\n"
        "     The wizard will generate a .env template at the end.\n"
        "     Fill in the key and re-run the wizard to generate prompts via LLM.\n"
    )
    skip: bool = questionary.confirm(
        "Skip LLM prompt generation and use example templates for now?",
        default=True,
    ).ask()
    return skip


def _copy_example_prompts(slug: str) -> tuple[Path, Path]:
    """Copy the built-in article prompts as a starting template."""
    gen_dst = _PROMPTS_DIR / f"wiki_summary_{slug}.md"
    eval_dst = _PROMPTS_DIR / f"wiki_evaluate_{slug}.md"
    if not gen_dst.exists():
        shutil.copy2(_PROMPTS_DIR / "wiki_summary_articles.md", gen_dst)
    if not eval_dst.exists():
        shutil.copy2(_PROMPTS_DIR / "wiki_evaluate_articles.md", eval_dst)
    return gen_dst, eval_dst


async def _generate_prompts_llm(
    slug: str,
    name: str,
    description: str,
    backend: str,
    model_id: str,
) -> tuple[Path, Path]:
    """Call the LLM to generate generate/evaluate prompts for a new entity type."""
    from .llm.factory import create_client  # noqa: PLC0415
    from .models.config import LLMConfig  # noqa: PLC0415

    llm = create_client(LLMConfig(backend=backend, model_id=model_id))  # type: ignore[arg-type]
    gen_dst = _PROMPTS_DIR / f"wiki_summary_{slug}.md"
    eval_dst = _PROMPTS_DIR / f"wiki_evaluate_{slug}.md"

    _SYS_GEN = (
        "You are an expert technical writer for wiki knowledge bases.\n"
        "Write a Jinja2 prompt template that instructs an LLM to generate a wiki "
        "summary page for a document of the given entity type.\n\n"
        "Available Jinja2 variables: {{ document_text }}, {{ language }}.\n"
        "Return only the prompt template — no explanation, no code fences."
    )
    _SYS_EVAL = (
        "You are an expert technical writer for wiki knowledge bases.\n"
        "Write a Jinja2 prompt template that instructs an LLM to evaluate the "
        "quality of a wiki page for the given entity type and return structured "
        "feedback (issues list + pass/fail verdict).\n\n"
        "Available Jinja2 variables: {{ wiki_page }}, {{ language }}.\n"
        "Return only the prompt template — no explanation, no code fences."
    )

    user = f"Entity type: {name}\nDescription: {description}"
    resp_gen = await llm.call(_SYS_GEN, user)
    resp_eval = await llm.call(_SYS_EVAL, user)

    gen_dst.write_text(resp_gen.text, encoding="utf-8")
    eval_dst.write_text(resp_eval.text, encoding="utf-8")
    return gen_dst, eval_dst


# ---------------------------------------------------------------------------
# Interactive steps
# ---------------------------------------------------------------------------


def _ask_entity_type(idx: int, backend: str, model_id: str, console: Console) -> dict:
    """Collect configuration for one entity type, including optional LLM prompt generation."""
    console.print(f"\n[bold cyan]── Entity Type #{idx} ──[/bold cyan]")

    name = questionary.text(
        "Display name (e.g. 'Internal Policy'):",
        validate=lambda v: bool(v.strip()) or "Required",
    ).ask()
    slug = questionary.text(
        "Slug (short unique id, e.g. 'policy'):",
        default=name.lower().replace(" ", "_"),
        validate=lambda v: bool(v.strip()) or "Required",
    ).ask()
    wiki_subdir = questionary.text(
        "Wiki subdirectory:", default=slug[:6]
    ).ask()
    frontmatter_raw = questionary.text(
        "Frontmatter fields to extract (comma-separated, or leave blank):",
        default="",
    ).ask()
    frontmatter_fields = [f.strip() for f in frontmatter_raw.split(",") if f.strip()]
    max_rounds = int(
        questionary.text(
            "Max writer→evaluator rounds:", default="2",
            validate=lambda v: v.isdigit() or "Must be an integer",
        ).ask()
    )

    gen_dst = _PROMPTS_DIR / f"wiki_summary_{slug}.md"
    eval_dst = _PROMPTS_DIR / f"wiki_evaluate_{slug}.md"

    if gen_dst.exists() and eval_dst.exists():
        console.print(f"[dim]  Prompts already exist for '{slug}', skipping generation.[/dim]")
    else:
        description = questionary.text(
            "Briefly describe this entity type (used to generate the prompts):",
            validate=lambda v: bool(v.strip()) or "Required",
        ).ask()

        use_llm = False
        if _env_key_present(backend):
            use_llm = questionary.confirm(
                f"Generate prompts via LLM ({model_id})?", default=True
            ).ask()
        else:
            use_llm = not _warn_no_env(backend, console)

        if use_llm:
            console.print("[dim]  Calling LLM to generate prompts…[/dim]")
            try:
                gen_dst, eval_dst = asyncio.run(
                    _generate_prompts_llm(slug, name, description, backend, model_id)
                )
                console.print(
                    f"[green]  ✓ Prompts saved:[/green] {gen_dst.name}, {eval_dst.name}"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"[red]  ✗ LLM failed ({exc}).[/red] Using example templates as fallback."
                )
                gen_dst, eval_dst = _copy_example_prompts(slug)
        else:
            gen_dst, eval_dst = _copy_example_prompts(slug)
            console.print(
                f"[dim]  Example templates copied as starting point: "
                f"{gen_dst.name}, {eval_dst.name}[/dim]"
            )

    return {
        "name": name,
        "slug": slug,
        "wiki_subdir": wiki_subdir,
        "prompt_generate": gen_dst,
        "prompt_evaluate": eval_dst,
        "frontmatter_fields": frontmatter_fields,
        "max_rounds": max_rounds,
    }


def _ask_taxonomy(idx: int, console: Console) -> dict:
    """Collect configuration for one taxonomy dimension."""
    console.print(f"\n[bold cyan]── Taxonomy #{idx} ──[/bold cyan]")

    name = questionary.text(
        "Taxonomy name (e.g. 'Topics'):",
        validate=lambda v: bool(v.strip()) or "Required",
    ).ask()
    wiki_subdir = questionary.text(
        "Wiki subdirectory:", default=name.lower().replace(" ", "_")
    ).ask()
    term_source = questionary.select(
        "How are terms extracted?",
        choices=[
            questionary.Choice("Wikilinks inside a Markdown section", value="section_wikilinks"),
            questionary.Choice("Frontmatter field value", value="metadata_field"),
        ],
    ).ask()

    section_header: str | None = None
    metadata_field: str | None = None
    if term_source == "section_wikilinks":
        section_header = questionary.text(
            "Section header to scan for [[wikilinks]] (e.g. '## Topic Connections'):",
            validate=lambda v: bool(v.strip()) or "Required",
        ).ask()
    else:
        metadata_field = questionary.text(
            "Frontmatter field name (e.g. 'theme'):",
            validate=lambda v: bool(v.strip()) or "Required",
        ).ask()

    # Copy normalize/create prompts if needed
    slug = name.lower().replace(" ", "_")
    norm_dst = _PROMPTS_DIR / f"wiki_{slug}_normalize.md"
    create_dst = _PROMPTS_DIR / f"wiki_agent_create_{slug}.md"
    if not norm_dst.exists():
        shutil.copy2(_PROMPTS_DIR / "wiki_themes_normalize.md", norm_dst)
    if not create_dst.exists():
        shutil.copy2(_PROMPTS_DIR / "wiki_agent_create_theme.md", create_dst)
    console.print(f"[dim]  Prompt templates: {norm_dst.name}, {create_dst.name}[/dim]")

    return {
        "name": name,
        "wiki_subdir": wiki_subdir,
        "term_source": term_source,
        "section_header": section_header,
        "metadata_field": metadata_field,
        "prompt_normalize": norm_dst,
        "prompt_create_page": create_dst,
    }


def _ask_grouping(idx: int) -> dict:
    """Collect configuration for one grouping dimension."""
    name = questionary.text(
        f"Grouping #{idx} name (e.g. 'Business Unit'):",
        validate=lambda v: bool(v.strip()) or "Required",
    ).ask()
    wiki_subdir = questionary.text(
        "Wiki subdirectory:", default=name.lower().replace(" ", "_")
    ).ask()
    metadata_field = questionary.text(
        "Document metadata field to group by (e.g. 'workspace', 'team'):",
        validate=lambda v: bool(v.strip()) or "Required",
    ).ask()
    return {"name": name, "wiki_subdir": wiki_subdir, "metadata_field": metadata_field}


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def _generate_config_py(
    wiki_name: str,
    wiki_dir: Path,
    content_dir: Path,
    log_dir: Path,
    language: str,
    backend: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    entity_types: list[dict],
    taxonomies: list[dict],
    groupings: list[dict],
) -> str:
    """Render the full content of my_wiki.py from wizard-collected data."""

    def _prompts_rel(p: Path) -> str:
        return "_PROMPTS / " + repr(p.name)

    lines: list[str] = [
        '"""wiki-llm configuration — generated by the setup wizard."""',
        "",
        "from __future__ import annotations",
        "",
        "import os",
        "from pathlib import Path",
        "",
        "from dotenv import load_dotenv",
        "",
        "from src.models.config import (",
        "    EntityTypeConfig,",
        "    GroupingConfig,",
        "    LLMConfig,",
        "    TaxonomyConfig,",
        "    WikiConfig,",
        ")",
        "from src.readers.base import MarkItDownPdfReader",
        "",
        "load_dotenv()",
        "",
        "# ---------------------------------------------------------------------------",
        "# Paths",
        "# ---------------------------------------------------------------------------",
        "",
        "_ROOT    = Path(__file__).parent.parent",
        "_PROMPTS = _ROOT / 'config' / 'prompts'",
        "",
    ]

    def _dir_expr(p: Path, env_var: str) -> str:
        try:
            rel = "/".join(p.relative_to(_ROOT).parts)
            return f"Path(os.environ.get({repr(env_var)}, str(_ROOT / {repr(rel)})))"
        except ValueError:
            return f"Path(os.environ.get({repr(env_var)}, {repr(str(p))}))"

    lines += [
        f"WIKI_DIR    = {_dir_expr(wiki_dir, 'WIKI_DIR')}",
        f"LOG_DIR     = {_dir_expr(log_dir, 'LOG_DIR')}",
        f"CONTENT_DIR = {_dir_expr(content_dir, 'CONTENT_DIR')}",
        "",
        "# ---------------------------------------------------------------------------",
        "# Main settings  (override via environment variables or edit directly)",
        "# ---------------------------------------------------------------------------",
        "",
        f"_BACKEND  = os.environ.get('WIKI_BACKEND',  {repr(backend)})",
        f"_MODEL_ID = os.environ.get('WIKI_MODEL_ID', {repr(model_id)})",
        f"_LANGUAGE = os.environ.get('WIKI_LANGUAGE', {repr(language)})",
        "",
        "# ---------------------------------------------------------------------------",
        "# WikiConfig",
        "# ---------------------------------------------------------------------------",
        "",
        "config = WikiConfig(",
        f"    wiki_name={repr(wiki_name)},",
        "    wiki_dir=WIKI_DIR,",
        "    log_dir=LOG_DIR,",
        "    language=_LANGUAGE,",
        "    llm=LLMConfig(",
        "        backend=_BACKEND,  # type: ignore[arg-type]",
        "        model_id=_MODEL_ID,",
        f"        temperature={temperature},",
        f"        max_tokens={max_tokens},",
        "    ),",
        "    entity_types=[",
    ]

    for et in entity_types:
        lines += [
            "        EntityTypeConfig(",
            f"            name={repr(et['name'])},",
            f"            slug={repr(et['slug'])},",
            f"            wiki_subdir={repr(et['wiki_subdir'])},",
            f"            prompt_generate={_prompts_rel(et['prompt_generate'])},",
            f"            prompt_evaluate={_prompts_rel(et['prompt_evaluate'])},",
            f"            frontmatter_fields={repr(et['frontmatter_fields'])},",
            f"            max_rounds={et['max_rounds']},",
            "        ),",
        ]
    lines.append("    ],")

    if taxonomies:
        lines.append("    taxonomies=[")
        for tx in taxonomies:
            lines += [
                "        TaxonomyConfig(",
                f"            name={repr(tx['name'])},",
                f"            wiki_subdir={repr(tx['wiki_subdir'])},",
                f"            term_source={repr(tx['term_source'])},",
            ]
            if tx["section_header"]:
                lines.append(f"            section_header={repr(tx['section_header'])},")
            if tx["metadata_field"]:
                lines.append(f"            metadata_field={repr(tx['metadata_field'])},")
            lines += [
                f"            prompt_normalize={_prompts_rel(tx['prompt_normalize'])},",
                f"            prompt_create_page={_prompts_rel(tx['prompt_create_page'])},",
                "        ),",
            ]
        lines.append("    ],")
    else:
        lines.append("    taxonomies=[],")

    if groupings:
        lines.append("    groupings=[")
        for grp in groupings:
            lines += [
                "        GroupingConfig(",
                f"            name={repr(grp['name'])},",
                f"            wiki_subdir={repr(grp['wiki_subdir'])},",
                f"            metadata_field={repr(grp['metadata_field'])},",
                "        ),",
            ]
        lines.append("    ],")
    else:
        lines.append("    groupings=[],")

    lines += [
        "    prompt_editor=_PROMPTS / 'wiki_editor.md',",
        "    prompt_lint=_PROMPTS / 'wiki_lint.md',",
        "    prompt_consolidate=_PROMPTS / 'wiki_consolidate_themes.md',",
        "    prompt_chat=_PROMPTS / 'wiki_chat.md',",
        "    status_filter=[],",
        "    max_chars_input=80_000,",
        "    on_llm_error='skip',",
        "    export_word=False,",
        "    content_dir=CONTENT_DIR,",
        "    pdf_reader=MarkItDownPdfReader(),",
        ")",
    ]

    return "\n".join(lines) + "\n"


def _generate_env(backend: str, model_id: str) -> str:
    """Generate a .env template with all variable labels but no secret values."""
    key = _BACKEND_KEYS.get(backend)
    lines = [
        "# wiki-llm — environment variables",
        "# Generated by the setup wizard. Fill in your credentials below.",
        "",
        f"WIKI_BACKEND={backend}",
        f"WIKI_MODEL_ID={model_id}",
        "# WIKI_LANGUAGE=english",
        "# WIKI_DIR=wiki",
        "# CONTENT_DIR=content_new",
        "# LOG_DIR=logs",
        "",
    ]

    if backend == "openrouter":
        lines += [
            "# OpenRouter API key — https://openrouter.ai/keys",
            f"{key}=",
        ]
    elif backend == "openai":
        lines += [
            "# OpenAI API key — https://platform.openai.com/api-keys",
            f"{key}=",
        ]
    elif backend == "bedrock":
        lines += [
            "# AWS credentials for Amazon Bedrock",
            f"{key}=",
            "AWS_ACCESS_KEY_ID=",
            "AWS_SECRET_ACCESS_KEY=",
            "AWS_DEFAULT_REGION=us-east-1",
        ]
    elif backend == "ollama":
        lines += [
            "# Ollama — no API key required",
            "# OLLAMA_BASE_URL=http://localhost:11434",
        ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_wizard() -> None:
    """Run the interactive setup wizard."""
    console = Console()

    # ── Welcome ──────────────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]{_LOGO}[/bold cyan]\n")
    console.print(Panel(
        "[bold white]Welcome to the wiki-llm setup wizard![/bold white]\n\n"
        "This wizard will guide you through configuring your wiki pipeline.\n"
        "It generates [bold]config/my_wiki.py[/bold] and a [bold].env[/bold] "
        "template with all required variable labels.\n\n"
        "[dim]Press Ctrl+C at any time to cancel without saving.[/dim]",
        border_style="cyan",
        expand=False,
    ))

    # ── Step 1 — Basic settings ───────────────────────────────────────────────
    console.print("\n[bold cyan]━━ Step 1 / 5  —  Basic Settings ━━[/bold cyan]")
    wiki_name   = questionary.text("Wiki name:", default="My Wiki").ask()
    wiki_dir    = Path(questionary.text("Wiki output directory:", default="wiki").ask())
    content_dir = Path(questionary.text("Content input directory:", default="content_new").ask())
    log_dir     = Path(questionary.text("Log directory:", default="logs").ask())
    language    = questionary.text(
        "Wiki language (e.g. 'english', 'português do Brasil'):", default="english"
    ).ask()

    # ── Step 2 — LLM ─────────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━ Step 2 / 5  —  LLM Configuration ━━[/bold cyan]")
    backend: str = questionary.select(
        "LLM backend:",
        choices=["openrouter", "openai", "bedrock", "ollama"],
        default="openrouter",
    ).ask()

    suggestions = _MODEL_SUGGESTIONS[backend]
    model_choice = questionary.select(
        "Model ID:",
        choices=[*suggestions, "[ enter manually ]"],
        default=suggestions[0],
    ).ask()
    model_id = (
        questionary.text("Model ID:", validate=lambda v: bool(v.strip()) or "Required").ask()
        if model_choice == "[ enter manually ]"
        else model_choice
    )

    temperature = float(
        questionary.text(
            "Temperature (0.0 – 1.0):", default="0.2",
            validate=lambda v: _is_float(v, 0.0, 1.0),
        ).ask()
    )
    max_tokens = int(
        questionary.text(
            "Max tokens:", default="4096",
            validate=lambda v: v.isdigit() or "Must be an integer",
        ).ask()
    )

    # ── Step 3 — Entity types ─────────────────────────────────────────────────
    console.print("\n[bold cyan]━━ Step 3 / 5  —  Entity Types ━━[/bold cyan]")
    console.print(
        "[dim]Entity types define the kinds of documents your wiki will contain "
        "(e.g. Policy, Article, Product, Report).[/dim]"
    )
    entity_types: list[dict] = []
    idx = 1
    while True:
        entity_types.append(_ask_entity_type(idx, backend, model_id, console))
        idx += 1
        if not questionary.confirm("Add another entity type?", default=False).ask():
            break

    # ── Step 4 — Taxonomies & Groupings ──────────────────────────────────────
    console.print("\n[bold cyan]━━ Step 4 / 5  —  Taxonomies & Groupings ━━[/bold cyan]")

    taxonomies: list[dict] = []
    if questionary.confirm(
        "Configure taxonomies? (topic/tag dimensions extracted from page content)",
        default=True,
    ).ask():
        t_idx = 1
        while True:
            taxonomies.append(_ask_taxonomy(t_idx, console))
            t_idx += 1
            if not questionary.confirm("Add another taxonomy?", default=False).ask():
                break

    groupings: list[dict] = []
    if questionary.confirm(
        "Configure groupings? (pages grouped by a document metadata field)",
        default=True,
    ).ask():
        g_idx = 1
        while True:
            groupings.append(_ask_grouping(g_idx))
            g_idx += 1
            if not questionary.confirm("Add another grouping?", default=False).ask():
                break

    # ── Step 5 — Write outputs ────────────────────────────────────────────────
    console.print("\n[bold cyan]━━ Step 5 / 5  —  Generating Files ━━[/bold cyan]")

    # Resolve paths relative to project root
    def _abs(p: Path) -> Path:
        return _ROOT / p if not p.is_absolute() else p

    wiki_dir_abs    = _abs(wiki_dir)
    content_dir_abs = _abs(content_dir)
    log_dir_abs     = _abs(log_dir)

    # Backup and write config/my_wiki.py
    if _CONFIG_PATH.exists():
        bak = _CONFIG_PATH.with_suffix(".py.bak")
        shutil.copy2(_CONFIG_PATH, bak)
        console.print(f"[dim]  Backup created: config/my_wiki.py.bak[/dim]")

    config_content = _generate_config_py(
        wiki_name=wiki_name,
        wiki_dir=wiki_dir_abs,
        content_dir=content_dir_abs,
        log_dir=log_dir_abs,
        language=language,
        backend=backend,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        entity_types=entity_types,
        taxonomies=taxonomies,
        groupings=groupings,
    )
    _CONFIG_PATH.write_text(config_content, encoding="utf-8")
    console.print("[green]  ✓ config/my_wiki.py[/green]")

    # Backup and write .env
    if _ENV_PATH.exists():
        shutil.copy2(_ENV_PATH, _ENV_PATH.with_name(".env.bak"))
        console.print("[dim]  Backup created: .env.bak[/dim]")
    _ENV_PATH.write_text(_generate_env(backend, model_id), encoding="utf-8")
    console.print("[green]  ✓ .env[/green]")

    # ── Summary ───────────────────────────────────────────────────────────────
    creds_label = _BACKEND_KEYS.get(backend) or "credentials"
    console.print(Panel(
        "[bold green]Setup complete![/bold green]\n\n"
        f"  Wiki      : {wiki_name}\n"
        f"  Backend   : {backend}  /  {model_id}\n"
        f"  Entities  : {len(entity_types)}\n"
        f"  Taxonomies: {len(taxonomies)}\n"
        f"  Groupings : {len(groupings)}\n\n"
        "[bold yellow]Next steps:[/bold yellow]\n"
        f"  1. Open [bold].env[/bold] and set [bold]{creds_label}[/bold]\n"
        "  2. Place source documents in [bold]content_new/[/bold]\n"
        "  3. Run the pipeline:\n"
        "     [bold cyan]uv run wiki-llm run-all --config config/my_wiki.py[/bold cyan]",
        border_style="green",
        expand=False,
    ))
