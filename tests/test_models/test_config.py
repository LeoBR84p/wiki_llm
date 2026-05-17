"""Unit tests for src/models/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.config import (
    GroupConfig,
    KeyThemeConfig,
    LLMConfig,
    ObjectTypeConfig,
    WikiConfig,
)


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_valid_openrouter(self):
        cfg = LLMConfig(backend="openrouter", model_id="anthropic/claude-3")
        assert cfg.backend == "openrouter"
        assert cfg.model_id == "anthropic/claude-3"

    def test_defaults(self):
        cfg = LLMConfig(backend="openai", model_id="gpt-4o")
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 4096

    def test_custom_temperature_and_tokens(self):
        cfg = LLMConfig(backend="ollama", model_id="llama3", temperature=0.7, max_tokens=512)
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 512

    def test_all_valid_backends(self):
        for backend in ("bedrock", "openrouter", "openai", "ollama"):
            cfg = LLMConfig(backend=backend, model_id="test")
            assert cfg.backend == backend

    def test_invalid_backend_raises(self):
        with pytest.raises(ValidationError):
            LLMConfig(backend="unknown_backend", model_id="test")

    def test_model_id_required(self):
        with pytest.raises(ValidationError):
            LLMConfig(backend="openai")


# ---------------------------------------------------------------------------
# ObjectTypeConfig
# ---------------------------------------------------------------------------


class TestObjectTypeConfig:
    def test_minimal(self, tmp_path):
        p = tmp_path / "gen.md"
        p.write_text("prompt")
        e = tmp_path / "eval.md"
        e.write_text("prompt")
        obj = ObjectTypeConfig(
            name="Policy",
            slug="policy",
            wiki_subdir="policies",
            prompt_generate=p,
            prompt_evaluate=e,
        )
        assert obj.slug == "policy"
        assert obj.max_rounds == 2
        assert obj.frontmatter_fields == []

    def test_custom_max_rounds(self, tmp_path):
        p = tmp_path / "g.md"
        p.write_text("x")
        e = tmp_path / "ev.md"
        e.write_text("x")
        obj = ObjectTypeConfig(
            name="Report",
            slug="report",
            wiki_subdir="reports",
            prompt_generate=p,
            prompt_evaluate=e,
            max_rounds=5,
            frontmatter_fields=["author", "date"],
        )
        assert obj.max_rounds == 5
        assert "author" in obj.frontmatter_fields

    def test_required_fields_missing(self):
        with pytest.raises(ValidationError):
            ObjectTypeConfig(name="X", slug="x")  # missing wiki_subdir and prompts


# ---------------------------------------------------------------------------
# KeyThemeConfig
# ---------------------------------------------------------------------------


class TestKeyThemeConfig:
    def test_section_wikilinks_valid(self, tmp_path):
        n = tmp_path / "norm.md"
        n.write_text("prompt")
        c = tmp_path / "create.md"
        c.write_text("prompt")
        kth = KeyThemeConfig(
            name="Topics",
            wiki_subdir="topics",
            term_source="section_wikilinks",
            section_header="## Topics",
            prompt_normalize=n,
            prompt_create_page=c,
        )
        assert kth.section_header == "## Topics"

    def test_section_wikilinks_missing_header_raises(self, tmp_path):
        n = tmp_path / "norm.md"
        n.write_text("prompt")
        c = tmp_path / "create.md"
        c.write_text("prompt")
        with pytest.raises(ValidationError, match="section_header is required"):
            KeyThemeConfig(
                name="Topics",
                wiki_subdir="topics",
                term_source="section_wikilinks",
                prompt_normalize=n,
                prompt_create_page=c,
            )

    def test_metadata_field_valid(self, tmp_path):
        n = tmp_path / "norm.md"
        n.write_text("prompt")
        c = tmp_path / "create.md"
        c.write_text("prompt")
        kth = KeyThemeConfig(
            name="Tags",
            wiki_subdir="tags",
            term_source="metadata_field",
            metadata_field="tags",
            prompt_normalize=n,
            prompt_create_page=c,
        )
        assert kth.metadata_field == "tags"

    def test_metadata_field_missing_raises(self, tmp_path):
        n = tmp_path / "norm.md"
        n.write_text("prompt")
        c = tmp_path / "create.md"
        c.write_text("prompt")
        with pytest.raises(ValidationError, match="metadata_field is required"):
            KeyThemeConfig(
                name="Tags",
                wiki_subdir="tags",
                term_source="metadata_field",
                prompt_normalize=n,
                prompt_create_page=c,
            )


# ---------------------------------------------------------------------------
# GroupConfig
# ---------------------------------------------------------------------------


class TestGroupConfig:
    def test_minimal(self):
        grp = GroupConfig(name="Team", wiki_subdir="teams", metadata_field="team")
        assert grp.name == "Team"
        assert grp.prompt_create_page is None

    def test_with_prompt(self, tmp_path):
        p = tmp_path / "grp.md"
        p.write_text("prompt")
        grp = GroupConfig(name="Dept", wiki_subdir="depts", metadata_field="dept", prompt_create_page=p)
        assert grp.prompt_create_page == p


# ---------------------------------------------------------------------------
# WikiConfig
# ---------------------------------------------------------------------------


class TestWikiConfig:
    def _make_prompts(self, root: Path) -> dict[str, Path]:
        prompts = root / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        for name in ["gen.md", "eval.md", "editor.md", "lint.md", "consol.md", "chat.md"]:
            (prompts / name).write_text("prompt", encoding="utf-8")
        return {
            "gen": prompts / "gen.md",
            "eval": prompts / "eval.md",
            "editor": prompts / "editor.md",
            "lint": prompts / "lint.md",
            "consol": prompts / "consol.md",
            "chat": prompts / "chat.md",
        }

    def _base_cfg(self, tmp_path: Path) -> dict:
        p = self._make_prompts(tmp_path)
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        logs = tmp_path / "logs"
        logs.mkdir()
        return dict(
            wiki_name="My Wiki",
            wiki_dir=wiki,
            log_dir=logs,
            llm=LLMConfig(backend="openai", model_id="gpt-4o"),
            objects=[
                ObjectTypeConfig(
                    name="Article",
                    slug="article",
                    wiki_subdir="articles",
                    prompt_generate=p["gen"],
                    prompt_evaluate=p["eval"],
                )
            ],
            prompt_editor=p["editor"],
            prompt_lint=p["lint"],
            prompt_consolidate=p["consol"],
            prompt_chat=p["chat"],
        )

    def test_minimal_valid(self, tmp_path):
        cfg = WikiConfig(**self._base_cfg(tmp_path))
        assert cfg.wiki_name == "My Wiki"
        assert cfg.language == "english"
        assert cfg.max_chars_input == 80_000

    def test_defaults(self, tmp_path):
        cfg = WikiConfig(**self._base_cfg(tmp_path))
        assert cfg.status_filter == []
        assert cfg.key_themes == []
        assert cfg.groups == []
        assert cfg.on_llm_error == "skip"
        assert cfg.export_word is False

    def test_empty_objects_raises(self, tmp_path):
        kwargs = self._base_cfg(tmp_path)
        kwargs["objects"] = []
        with pytest.raises(ValidationError, match="objects cannot be empty"):
            WikiConfig(**kwargs)

    def test_duplicate_slugs_raises(self, tmp_path):
        kwargs = self._base_cfg(tmp_path)
        p = self._make_prompts(tmp_path / "extra")
        obj_dup = ObjectTypeConfig(
            name="Other",
            slug="article",  # same slug as the first
            wiki_subdir="other",
            prompt_generate=p["gen"],
            prompt_evaluate=p["eval"],
        )
        kwargs["objects"].append(obj_dup)
        with pytest.raises(ValidationError, match="slugs must be unique"):
            WikiConfig(**kwargs)

    def test_object_by_slug_found(self, tmp_path):
        cfg = WikiConfig(**self._base_cfg(tmp_path))
        obj = cfg.object_by_slug("article")
        assert obj is not None
        assert obj.name == "Article"

    def test_object_by_slug_not_found(self, tmp_path):
        cfg = WikiConfig(**self._base_cfg(tmp_path))
        assert cfg.object_by_slug("nonexistent") is None

    def test_content_dir_default(self, tmp_path):
        cfg = WikiConfig(**self._base_cfg(tmp_path))
        assert cfg.content_dir == Path("content_new")

    def test_custom_language(self, tmp_path):
        kwargs = self._base_cfg(tmp_path)
        kwargs["language"] = "portuguese"
        cfg = WikiConfig(**kwargs)
        assert cfg.language == "portuguese"
