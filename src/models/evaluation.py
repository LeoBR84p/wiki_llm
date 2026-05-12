"""Pydantic models for wiki page evaluation and repair tracking.

Used as structured extraction schemas when calling the LLM evaluator,
and as the state object passed from the lint stage to the repair stage.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageEvaluation(BaseModel):
    """Structured LLM output from the page evaluator node.

    Extracted via instructor from the evaluator prompt response.
    The generate stage loops writer→evaluator→editor until
    approved is True or max_rounds is reached.

    Attributes:
        approved: True if the draft is suitable for publication.
        problems: List of specific problems found in the draft.
        suggestions: List of improvement suggestions for the editor.
    """
    approved: bool = Field(description="True se o rascunho está adequado para publicação")
    problems: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class RepairState(BaseModel):
    """State object produced by the lint stage and consumed by the repair stage.

    Attributes:
        wiki_dir: Absolute path of the wiki root directory as a string.
        orphans: Page stems that have no inbound wikilinks.
        broken_links: List of {origem, destino} dicts for broken wikilinks.
        repaired: Page stems successfully repaired by the repair agent.
        errors: Error strings for items that could not be repaired.
    """
    wiki_dir: str
    orphans: list[str] = Field(default_factory=list)
    broken_links: list[dict[str, Any]] = Field(default_factory=list)
    repaired: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
