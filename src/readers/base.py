"""Reader protocols for document ingestion.

Defines the structural interfaces (Protocols) that all reader implementations
must satisfy.  The filesystem reader and any user-supplied PDF reader both
conform to these protocols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models.document import Document


class BaseReader(Protocol):
    """Protocol for document readers that can asynchronously return a list of Documents.

    Any class implementing this protocol can be used as the primary content
    reader in the pipeline.  The filesystem reader satisfies this interface.
    """
    async def read_all(self) -> list[Document]: ...


@runtime_checkable
class PdfReaderProtocol(Protocol):
    """Protocol for extracting plain text from PDF files.

    Implementations:
      PymupdfReader   -- digital text extraction, no OCR required
      TesseractReader -- OCR via pytesseract + pdf2image
      AzureDIReader   -- Azure Document Intelligence API

    Used by the filesystem reader when a .pdf file is encountered and
    a pdf_reader is configured in WikiConfig.
    """

    def extract_text(self, path: Path) -> str: ...
