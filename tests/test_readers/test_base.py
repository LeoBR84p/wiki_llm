"""Unit tests for src/readers/base.py — protocols and MarkItDownPdfReader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.readers.base import BaseReader, MarkItDownPdfReader, PdfReaderProtocol


class TestProtocols:
    def test_pdf_reader_protocol_isinstance_passes(self):
        class FakeReader:
            def extract_text(self, path: Path) -> str:
                return "text"

        assert isinstance(FakeReader(), PdfReaderProtocol)

    def test_pdf_reader_protocol_missing_method_fails(self):
        class BadReader:
            pass

        assert not isinstance(BadReader(), PdfReaderProtocol)

    def test_base_reader_is_protocol(self):
        # BaseReader is a structural Protocol
        import inspect
        from typing import Protocol
        assert issubclass(BaseReader, Protocol)

    def test_markitdown_reader_satisfies_protocol(self):
        reader = MarkItDownPdfReader()
        assert isinstance(reader, PdfReaderProtocol)


class TestMarkItDownPdfReader:
    def test_extract_text_calls_markitdown(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_instance = MagicMock()
        mock_instance.convert.return_value = MagicMock(text_content="Extracted text")
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict("sys.modules", {"markitdown": MagicMock(MarkItDown=mock_cls)}):
            reader = MarkItDownPdfReader()
            result = reader.extract_text(pdf_path)

        assert result == "Extracted text"

    def test_extract_text_returns_empty_when_none(self, tmp_path):
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"empty")

        mock_instance = MagicMock()
        mock_instance.convert.return_value = MagicMock(text_content=None)
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict("sys.modules", {"markitdown": MagicMock(MarkItDown=mock_cls)}):
            reader = MarkItDownPdfReader()
            result = reader.extract_text(pdf_path)

        assert result == ""

    def test_extract_text_returns_empty_string_for_empty_content(self, tmp_path):
        pdf_path = tmp_path / "blank.pdf"
        pdf_path.write_bytes(b"blank")

        mock_instance = MagicMock()
        mock_instance.convert.return_value = MagicMock(text_content="")
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict("sys.modules", {"markitdown": MagicMock(MarkItDown=mock_cls)}):
            reader = MarkItDownPdfReader()
            result = reader.extract_text(pdf_path)

        assert result == ""
