"""Configuração de exemplo — equivalente funcional ao EXAMPLE/wiki_ng/.

Replica a estrutura do sistema de normas (NI + NE + Temas + Unidades) usando
a nova API genérica de src/.

Uso:
    python -m src --config config/exemplo_normas.py tudo

Variáveis de ambiente necessárias (ver .env.example):
    WIKI_BACKEND         — bedrock | openrouter | openai
    WIKI_MODEL_ID        — modelo padrão
    OPENROUTER_APIKEY    — se backend = openrouter
    AWS_LOGINKEY         — se backend = bedrock

Para testes gratuitos use OpenRouter com modelos free, ex:
    WIKI_BACKEND=openrouter
    WIKI_MODEL_ID=mistralai/mistral-7b-instruct:free
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

_ni = EntityTypeConfig(
    name="Norma Interna",
    slug="ni",
    wiki_subdir="NI",
    prompt_generate=_PROMPTS / "wiki_resumo.md",
    prompt_evaluate=_PROMPTS / "wiki_avaliador.md",
    frontmatter_fields=["area_gestora", "vigencia", "assunto"],
    max_rounds=2,
)

_ne = EntityTypeConfig(
    name="Norma Externa",
    slug="ne",
    wiki_subdir="NE",
    prompt_generate=_PROMPTS / "wiki_resumo_ne.md",
    prompt_evaluate=_PROMPTS / "wiki_avaliador.md",
    frontmatter_fields=["orgao_emissor", "vigencia", "assunto"],
    max_rounds=2,
)

# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------

_temas = TaxonomyConfig(
    name="Temas",
    wiki_subdir="Temas",
    section_header="## Conexões Temáticas",
    prompt_normalize=_PROMPTS / "wiki_temas_normalizar.md",
    prompt_create_page=_PROMPTS / "wiki_agente_criar_tema.md",
)

# ---------------------------------------------------------------------------
# Groupings (organizational units)
# ---------------------------------------------------------------------------

_unidades = GroupingConfig(
    name="Unidades",
    wiki_subdir="Unidades",
    metadata_field="area_gestora",
)

# ---------------------------------------------------------------------------
# WikiConfig
# ---------------------------------------------------------------------------

config = WikiConfig(
    wiki_name="Wiki de Normas",
    wiki_dir=WIKI_DIR,
    log_dir=LOG_DIR,
    llm=llm_cfg,
    entity_types=[_ni, _ne],
    taxonomies=[_temas],
    groupings=[_unidades],
    # Prompts comuns
    prompt_editor=_PROMPTS / "wiki_editor.md",
    prompt_lint=_PROMPTS / "wiki_lint.md",
    prompt_consolidate=_PROMPTS / "wiki_consolidar_temas.md",
    prompt_chat=_PROMPTS / "wiki_chat.md",
    # Comportamento
    status_filter=["vigente", "em_vigor", "ativo", ""],
    max_chars_input=80_000,
    on_llm_error="skip",
    export_word=False,
    content_dir=CONTENT_DIR,
)
