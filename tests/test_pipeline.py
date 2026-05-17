"""Unit tests for src/pipeline.py — dataclasses and option defaults."""

from __future__ import annotations

import pytest

from src.pipeline import PipelineOptions, PipelineResult


class TestPipelineOptions:
    def test_defaults(self):
        opts = PipelineOptions()
        assert opts.force is False
        assert opts.workers == 4
        assert "read" in opts.stages
        assert "generate" in opts.stages
        assert "lint" in opts.stages
        assert "repair" in opts.stages

    def test_all_default_stages_present(self):
        opts = PipelineOptions()
        expected = {"read", "generate", "key_themes", "groups", "index", "consolidate", "lint", "repair"}
        assert expected.issubset(set(opts.stages))

    def test_custom_force(self):
        opts = PipelineOptions(force=True)
        assert opts.force is True

    def test_custom_workers(self):
        opts = PipelineOptions(workers=8)
        assert opts.workers == 8

    def test_custom_stages(self):
        opts = PipelineOptions(stages=["read", "generate"])
        assert opts.stages == ["read", "generate"]
        assert "lint" not in opts.stages

    def test_empty_stages(self):
        opts = PipelineOptions(stages=[])
        assert opts.stages == []


class TestPipelineResult:
    def test_defaults(self):
        result = PipelineResult()
        assert result.docs_read == 0
        assert result.pages_generated == 0
        assert result.pages_error == 0
        assert result.elapsed_s == 0.0

    def test_can_set_fields(self):
        result = PipelineResult(
            docs_read=10,
            pages_generated=8,
            pages_error=1,
            elapsed_s=42.5,
        )
        assert result.docs_read == 10
        assert result.pages_generated == 8
        assert result.pages_error == 1
        assert result.elapsed_s == 42.5

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(PipelineResult)


# ---------------------------------------------------------------------------
# run_pipeline (async, with all stages mocked)
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path):
    from src.models.config import LLMConfig, ObjectTypeConfig, WikiConfig
    wiki = tmp_path / "wiki"; wiki.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    prompts = tmp_path / "prompts"; prompts.mkdir()
    for fn in ["gen.md", "eval.md", "editor.md", "lint.md", "consol.md", "chat.md"]:
        (prompts / fn).write_text("prompt text")
    return WikiConfig(
        wiki_name="Test", wiki_dir=wiki, log_dir=logs,
        llm=LLMConfig(backend="openrouter", model_id="test"),
        objects=[ObjectTypeConfig(
            name="Article", slug="article", wiki_subdir="articles",
            prompt_generate=prompts / "gen.md",
            prompt_evaluate=prompts / "eval.md",
        )],
        prompt_editor=prompts / "editor.md",
        prompt_lint=prompts / "lint.md",
        prompt_consolidate=prompts / "consol.md",
        prompt_chat=prompts / "chat.md",
    )


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_run_pipeline_no_stages(self, tmp_path):
        from src.pipeline import run_pipeline, PipelineOptions
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_cfg(tmp_path)
        opts = PipelineOptions(stages=[])

        with patch("src.pipeline.create_client", return_value=MagicMock()), \
             patch("src.pipeline.LLMLogger", return_value=MagicMock()):
            result = await run_pipeline(cfg, opts)

        assert result.docs_read == 0
        assert result.pages_generated == 0
        assert result.elapsed_s >= 0.0

    @pytest.mark.asyncio
    async def test_run_pipeline_read_only(self, tmp_path):
        from src.pipeline import run_pipeline, PipelineOptions
        from src.models.document import Document
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_cfg(tmp_path)
        opts = PipelineOptions(stages=["read"])

        mock_doc = MagicMock(spec=Document)
        mock_doc.metadata = MagicMock(status=None, id="test-doc")

        with patch("src.pipeline.create_client", return_value=MagicMock()), \
             patch("src.pipeline.LLMLogger", return_value=MagicMock()), \
             patch("src.pipeline.FilesystemReader") as MockReader:
            mock_reader_inst = MagicMock()
            mock_reader_inst.read_all = AsyncMock(return_value=[mock_doc])
            MockReader.return_value = mock_reader_inst
            result = await run_pipeline(cfg, opts)

        assert result.docs_read == 1

    @pytest.mark.asyncio
    async def test_run_pipeline_index_only(self, tmp_path):
        from src.pipeline import run_pipeline, PipelineOptions
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_cfg(tmp_path)
        opts = PipelineOptions(stages=["index"])

        with patch("src.pipeline.create_client", return_value=MagicMock()), \
             patch("src.pipeline.LLMLogger", return_value=MagicMock()), \
             patch("src.pipeline.run_index", new=AsyncMock()):
            result = await run_pipeline(cfg, opts)

        assert result.elapsed_s >= 0.0

    @pytest.mark.asyncio
    async def test_run_pipeline_consolidate_lint_stages(self, tmp_path):
        from src.pipeline import run_pipeline, PipelineOptions
        from src.models.evaluation import RepairState
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_cfg(tmp_path)
        opts = PipelineOptions(stages=["consolidate", "lint"])

        repair_state = RepairState(
            wiki_dir=cfg.wiki_dir,
            broken_links=[],
            orphans=[],
            evaluations=[],
        )

        with patch("src.pipeline.create_client", return_value=MagicMock()), \
             patch("src.pipeline.LLMLogger", return_value=MagicMock()), \
             patch("src.pipeline.run_consolidate", new=AsyncMock()), \
             patch("src.pipeline.run_lint", new=AsyncMock(return_value=repair_state)):
            result = await run_pipeline(cfg, opts)

        assert result.elapsed_s >= 0.0

    @pytest.mark.asyncio
    async def test_run_pipeline_repair_stage(self, tmp_path):
        from src.pipeline import run_pipeline, PipelineOptions
        from src.models.evaluation import RepairState
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_cfg(tmp_path)
        opts = PipelineOptions(stages=["lint", "repair"])

        repair_state = RepairState(
            wiki_dir=cfg.wiki_dir,
            broken_links=[],
            orphans=[],
            evaluations=[],
        )

        with patch("src.pipeline.create_client", return_value=MagicMock()), \
             patch("src.pipeline.LLMLogger", return_value=MagicMock()), \
             patch("src.pipeline.run_lint", new=AsyncMock(return_value=repair_state)), \
             patch("src.pipeline.run_repair", new=AsyncMock(return_value=repair_state)):
            result = await run_pipeline(cfg, opts)

        assert result.elapsed_s >= 0.0

    @pytest.mark.asyncio
    async def test_run_pipeline_generate_stage_counts_errors(self, tmp_path):
        from src.pipeline import run_pipeline, PipelineOptions
        from src.models.document import Document
        from unittest.mock import AsyncMock, MagicMock, patch

        cfg = _make_cfg(tmp_path)
        opts = PipelineOptions(stages=["read", "generate"], workers=1)

        mock_doc = MagicMock(spec=Document)
        mock_doc.metadata = MagicMock(status=None, id="doc-1")

        with patch("src.pipeline.create_client", return_value=MagicMock()), \
             patch("src.pipeline.LLMLogger", return_value=MagicMock()), \
             patch("src.pipeline.FilesystemReader") as MockReader, \
             patch("src.pipeline.generate_page", new=AsyncMock(side_effect=RuntimeError("fail"))):
            mock_reader_inst = MagicMock()
            mock_reader_inst.read_all = AsyncMock(return_value=[mock_doc])
            MockReader.return_value = mock_reader_inst
            result = await run_pipeline(cfg, opts)

        assert result.docs_read == 1
        assert result.pages_error == 1

    @pytest.mark.asyncio
    async def test_run_pipeline_returns_default_opts_when_none(self, tmp_path):
        from src.pipeline import run_pipeline
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.models.evaluation import RepairState

        cfg = _make_cfg(tmp_path)
        repair_state = RepairState(
            wiki_dir=cfg.wiki_dir, broken_links=[], orphans=[], evaluations=[],
        )

        with patch("src.pipeline.create_client", return_value=MagicMock()), \
             patch("src.pipeline.LLMLogger", return_value=MagicMock()), \
             patch("src.pipeline.FilesystemReader") as MockReader, \
             patch("src.pipeline.run_key_themes", new=AsyncMock()), \
             patch("src.pipeline.run_groups", new=AsyncMock()), \
             patch("src.pipeline.run_index", new=AsyncMock()), \
             patch("src.pipeline.run_consolidate", new=AsyncMock()), \
             patch("src.pipeline.run_lint", new=AsyncMock(return_value=repair_state)), \
             patch("src.pipeline.run_repair", new=AsyncMock(return_value=repair_state)):
            mock_reader_inst = MagicMock()
            mock_reader_inst.read_all = AsyncMock(return_value=[])
            MockReader.return_value = mock_reader_inst
            result = await run_pipeline(cfg, opts=None)

        assert result.docs_read == 0
        assert result.elapsed_s >= 0.0
