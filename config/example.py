"""Configuração de exemplo.

Uso:
    python -m src --config config/exemplo.py tudo

Variáveis de ambiente necessárias (ver .env.example):
    WIKI_BACKEND         — bedrock | openrouter | openai
    WIKI_MODEL_ID        — modelo padrão
    OPENROUTER_APIKEY    — se backend = openrouter
    AWS_LOGINKEY         — se backend = bedrock

Para testes gratuitos use OpenRouter com modelos free, ex:
    WIKI_BACKEND=openrouter
    WIKI_MODEL_ID=google/gemma-4-26b-a4b-it:free
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.models.config import (
    EntityTypeConfig,
    GroupingConfig,
    LLMConfig,
    TaxonomyConfig,
    WikiConfig,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent  # c:\projetos\wiki_llm

# Reutiliza prompts do EXAMPLE/ — copie-os para config/prompts/ para personalizar
_PROMPTS = _ROOT / "EXAMPLE" / "wiki_ng" / "prompts"

# Destino da wiki gerada
WIKI_DIR = Path(os.environ.get("WIKI_DIR", str(_ROOT / "wiki")))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(_ROOT / "logs")))
CONTENT_DIR = Path(os.environ.get("CONTENT_DIR", str(_ROOT / "content_new")))

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("WIKI_BACKEND", "openrouter")
_MODEL_ID = os.environ.get("WIKI_MODEL_ID", "mistralai/mistral-7b-instruct:free")

llm_cfg = LLMConfig(
    backend=_BACKEND,  # type: ignore[arg-type]
    model_id=_MODEL_ID,
    temperature=0.2,
    max_tokens=4096,
)

# ---------------------------------------------------------------------------
# Entity Types
# ---------------------------------------------------------------------------

_arts = EntityTypeConfig(
    name="My Articles",
    slug="articles",
    wiki_subdir="arts",
    prompt_generate=_PROMPTS / "wiki_summary_articles.md",
    prompt_evaluate=_PROMPTS / "wiki_evaluate_articles.md",
    frontmatter_fields=["theme", "platform"],
    max_rounds=2,
)

_proj = EntityTypeConfig(
    name="My Projects",
    slug="projects",
    wiki_subdir="proj",
    prompt_generate=_PROMPTS / "wiki_summary_projects.md",
    prompt_evaluate=_PROMPTS / "wiki_evaluate_projects.md",
    frontmatter_fields=["theme", "articles", "technology_subject"],
    max_rounds=2,
)

# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------

_themes = TaxonomyConfig(
    name="Themes",
    wiki_subdir="themes",
    section_header="## Theme Connections",
    prompt_normalize=_PROMPTS / "wiki_themes_normalize.md",
    prompt_create_page=_PROMPTS / "wiki_agent_create_theme.md",
)

# ---------------------------------------------------------------------------
# Groupings (workspaces)
# ---------------------------------------------------------------------------

_workspaces = GroupingConfig(
    name="Workspace",
    wiki_subdir="workspaces",
    metadata_field="workspace",
)

# ---------------------------------------------------------------------------
# WikiConfig
# ---------------------------------------------------------------------------

config = WikiConfig(
    wiki_name="My Wiki",
    wiki_dir=WIKI_DIR,
    log_dir=LOG_DIR,
    llm=llm_cfg,
    entity_types=[_arts, _proj],
    taxonomies=[_themes],
    groupings=[_workspaces],
    # Prompts comuns
    prompt_editor=_PROMPTS / "wiki_editor.md",
    prompt_lint=_PROMPTS / "wiki_lint.md",
    prompt_consolidate=_PROMPTS / "wiki_consolidate_themes.md",
    prompt_chat=_PROMPTS / "wiki_chat.md",
    # Comportamento
    status_filter=["staged", "production", ""],
    max_chars_input=80_000,
    on_llm_error="skip",
    export_word=False,
    content_dir=CONTENT_DIR,
)
