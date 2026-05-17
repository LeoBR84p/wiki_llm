"""Shared pytest fixtures for the wiki-llm test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.config import (
    GroupConfig,
    KeyThemeConfig,
    LLMConfig,
    ObjectTypeConfig,
    WikiConfig,
)
from src.models.document import Document, DocumentMetadata


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _make_prompts(root: Path) -> dict[str, Path]:
    """Write minimal prompt files and return a name→path dict."""
    prompts = root / "prompts"
    prompts.mkdir(exist_ok=True)
    files = {
        "generate": "# Writer prompt\n{{ document }}",
        "evaluate": "# Evaluator prompt\n{{ draft }}",
        "editor": "# Editor prompt\n{{ draft }}",
        "lint": "# Lint prompt",
        "consolidate": "# Consolidate prompt",
        "chat": "# Chat prompt",
        "normalize": "# Normalize prompt",
        "theme_create": "# Theme create prompt",
        "group_create": "# Group create prompt",
    }
    paths: dict[str, Path] = {}
    for name, content in files.items():
        p = prompts / f"{name}.md"
        p.write_text(content, encoding="utf-8")
        paths[name] = p
    return paths


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create a standard directory layout for tests."""
    dirs = {
        "root": tmp_path,
        "wiki": tmp_path / "wiki",
        "logs": tmp_path / "logs",
        "content": tmp_path / "content",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture()
def prompts(tmp_path: Path) -> dict[str, Path]:
    return _make_prompts(tmp_path)


@pytest.fixture()
def minimal_config(tmp_dirs: dict[str, Path], prompts: dict[str, Path]) -> WikiConfig:
    """Minimal WikiConfig suitable for stage and pipeline tests."""
    return WikiConfig(
        wiki_name="Test Wiki",
        wiki_dir=tmp_dirs["wiki"],
        log_dir=tmp_dirs["logs"],
        content_dir=tmp_dirs["content"],
        llm=LLMConfig(backend="openrouter", model_id="test-model"),
        objects=[
            ObjectTypeConfig(
                name="Article",
                slug="article",
                wiki_subdir="articles",
                prompt_generate=prompts["generate"],
                prompt_evaluate=prompts["evaluate"],
            )
        ],
        prompt_editor=prompts["editor"],
        prompt_lint=prompts["lint"],
        prompt_consolidate=prompts["consolidate"],
        prompt_chat=prompts["chat"],
    )


@pytest.fixture()
def config_with_themes_and_groups(
    tmp_dirs: dict[str, Path], prompts: dict[str, Path]
) -> WikiConfig:
    """WikiConfig with key_themes and groups configured."""
    return WikiConfig(
        wiki_name="Full Wiki",
        wiki_dir=tmp_dirs["wiki"],
        log_dir=tmp_dirs["logs"],
        content_dir=tmp_dirs["content"],
        llm=LLMConfig(backend="openrouter", model_id="test-model"),
        objects=[
            ObjectTypeConfig(
                name="Article",
                slug="article",
                wiki_subdir="articles",
                prompt_generate=prompts["generate"],
                prompt_evaluate=prompts["evaluate"],
            )
        ],
        key_themes=[
            KeyThemeConfig(
                name="Topics",
                wiki_subdir="topics",
                term_source="section_wikilinks",
                section_header="## Topics",
                prompt_normalize=prompts["normalize"],
                prompt_create_page=prompts["theme_create"],
            )
        ],
        groups=[
            GroupConfig(
                name="Team",
                wiki_subdir="teams",
                metadata_field="team",
            )
        ],
        prompt_editor=prompts["editor"],
        prompt_lint=prompts["lint"],
        prompt_consolidate=prompts["consolidate"],
        prompt_chat=prompts["chat"],
    )


@pytest.fixture()
def sample_document(tmp_dirs: dict[str, Path]) -> Document:
    """A simple Document fixture for use in multiple tests."""
    content_file = tmp_dirs["content"] / "sample.md"
    content_file.write_text("# Sample\n\nBody text.", encoding="utf-8")
    return Document(
        metadata=DocumentMetadata(
            id="550e8400-e29b-41d4-a716-446655440000",
            title="Sample Document",
            object_type="article",
            status="active",
            extra={"team": "Engineering", "source_filename": "sample.md"},
        ),
        content="# Sample\n\nBody text.",
        content_path=content_file,
    )
