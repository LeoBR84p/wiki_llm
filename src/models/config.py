"""Pydantic configuration models for the wiki-llm pipeline.

All domain-specific behaviour is declared here by the user.
The generic pipeline in src/ accepts a WikiConfig and operates
without any domain-specific knowledge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """LLM backend selection and generation parameters.

    Attributes:
        backend: Which LLM provider to use.
        model_id: Model identifier string passed to the provider API.
        temperature: Sampling temperature (0 = deterministic, higher = more creative).
        max_tokens: Maximum tokens to generate per call.
    """
    backend: Literal["bedrock", "openrouter", "openai", "ollama"] = "openrouter"
    model_id: str
    temperature: float = 0.2
    max_tokens: int = 4096


# ---------------------------------------------------------------------------
# Entity types, taxonomies and groupings
# ---------------------------------------------------------------------------


class EntityTypeConfig(BaseModel):
    """Defines one entity type that the wiki can contain.

    Example entity types: Internal Policy, External Standard, Article, Product.
    Each entity type gets its own wiki subdirectory and its own generate/evaluate prompts.

    Attributes:
        name: Human-readable name displayed in the index (e.g. "Internal Policy").
        slug: Short unique identifier used in code (e.g. "internal_policy").
        wiki_subdir: Subdirectory name under wiki_dir for this entity type.
        prompt_generate: Path to the Jinja2 writer prompt file.
        prompt_evaluate: Path to the Jinja2 evaluator prompt file.
        frontmatter_fields: Metadata field names to extract into the page frontmatter.
        max_rounds: Maximum writer→evaluator→editor loop iterations.
    """

    name: str
    slug: str
    wiki_subdir: str
    prompt_generate: Path
    prompt_evaluate: Path
    frontmatter_fields: list[str] = Field(default_factory=list)
    max_rounds: int = 2


class TaxonomyConfig(BaseModel):
    """Defines a taxonomy dimension (e.g. Topics, Tags, Categories).

    Attributes:
        name: Human-readable taxonomy name (e.g. "Topics").
        wiki_subdir: Subdirectory under wiki_dir for taxonomy pages.
        term_source: Where to extract terms from — wikilinks inside a section
            header (``section_wikilinks``) or a frontmatter field value
            (``metadata_field``).
        section_header: Markdown section heading to scan for ``[[wikilinks]]``.
            Required when term_source is ``section_wikilinks``.
        metadata_field: Frontmatter field name whose value(s) become terms.
            Required when term_source is ``metadata_field``.
        prompt_normalize: Prompt file for batch LLM term normalization.
        prompt_create_page: Prompt file for generating taxonomy summary pages.
    """

    name: str
    wiki_subdir: str
    term_source: Literal["section_wikilinks", "metadata_field"] = "section_wikilinks"
    section_header: str | None = None
    metadata_field: str | None = None
    prompt_normalize: Path
    prompt_create_page: Path

    @model_validator(mode="after")
    def _check_source_fields(self) -> "TaxonomyConfig":
        if self.term_source == "section_wikilinks" and not self.section_header:
            raise ValueError("section_header is required when term_source='section_wikilinks'")
        if self.term_source == "metadata_field" and not self.metadata_field:
            raise ValueError("metadata_field is required when term_source='metadata_field'")
        return self


class GroupingConfig(BaseModel):
    """Defines an organizational grouping dimension (e.g. Business Units, Authors).

    Attributes:
        name: Human-readable grouping name (e.g. "Business Unit").
        wiki_subdir: Subdirectory under wiki_dir for grouping pages.
        metadata_field: The document metadata field used to group documents.
        prompt_create_page: Optional prompt file for an LLM-generated summary
            page.  When None, a mechanical Markdown table is produced instead.
    """

    name: str
    wiki_subdir: str
    metadata_field: str
    prompt_create_page: Path | None = None


# ---------------------------------------------------------------------------
# WikiConfig — central pipeline contract
# ---------------------------------------------------------------------------


class WikiConfig(BaseModel):
    """Central configuration contract for the wiki-llm pipeline.

    Users declare all domain-specific information here.  The pipeline in
    src/ receives a WikiConfig and operates without any domain knowledge.

    Attributes:
        wiki_name: Display name for the generated wiki.
        wiki_dir: Root directory where wiki pages are written.
        log_dir: Directory for LLM call logs (JSONL files).
        llm: LLM backend configuration.
        entity_types: Ordered list of entity type definitions (must be non-empty).
        taxonomies: Optional taxonomy dimensions.
        groupings: Optional organizational grouping dimensions.
        prompt_editor: Path to the editor prompt used in the generate stage.
        prompt_lint: Path to the lint/repair prompt.
        prompt_consolidate: Path to the consolidation prompt.
        prompt_chat: Path to the chat RAG prompt.
        status_filter: If non-empty, only documents with matching status are processed.
        max_chars_input: Maximum characters of document content passed to the LLM.
        batch_size: Number of items per LLM batch call.
        on_llm_error: Whether to skip or abort when an LLM call fails.
        export_word: Whether to export generated pages to .docx format.
        content_dir: Directory containing new source documents to ingest.
        content_processed_dir: Override destination for successfully processed files.
        content_error_dir: Override destination for files that failed processing.
        pdf_reader: Pluggable PDF reader implementing PdfReaderProtocol.
    """
    wiki_name: str
    wiki_dir: Path
    log_dir: Path
    language: str = "english"

    llm: LLMConfig

    entity_types: list[EntityTypeConfig]
    taxonomies: list[TaxonomyConfig] = Field(default_factory=list)
    groupings: list[GroupingConfig] = Field(default_factory=list)

    prompt_editor: Path
    prompt_lint: Path
    prompt_consolidate: Path
    prompt_chat: Path

    status_filter: list[str] = Field(default_factory=list)
    max_chars_input: int = 80_000
    batch_size: int = 80
    on_llm_error: Literal["skip", "abort"] = "skip"
    export_word: bool = False

    content_dir: Path = Path("content_new")
    content_processed_dir: Path | None = None
    content_error_dir: Path | None = None

    pdf_reader: Any | None = None

    def get_processed_dir(self) -> Path:
        """Return the directory for successfully processed source files.

        Returns:
            content_processed_dir if explicitly set, otherwise
            the ``content_processed`` sibling of content_dir.
        """
        return self.content_processed_dir or self.content_dir.parent / "content_processed"

    def get_error_dir(self) -> Path:
        """Return the directory for source files that failed processing.

        Returns:
            content_error_dir if explicitly set, otherwise
            the ``content_error`` sibling of content_dir.
        """
        return self.content_error_dir or self.content_dir.parent / "content_error"

    @model_validator(mode="after")
    def _check_entity_types(self) -> WikiConfig:
        if not self.entity_types:
            raise ValueError("entity_types cannot be empty.")
        slugs = [e.slug for e in self.entity_types]
        if len(slugs) != len(set(slugs)):
            raise ValueError("entity_types slugs must be unique.")
        return self

    def entity_type_by_slug(self, slug: str) -> EntityTypeConfig | None:
        for et in self.entity_types:
            if et.slug == slug:
                return et
        return None
