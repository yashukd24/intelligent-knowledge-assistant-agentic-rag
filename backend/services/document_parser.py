"""
backend/services/document_parser.py
─────────────────────────────────────────────────────────────
Document Text Extraction Service for PDF, DOCX, and TXT files.

Extracts text content and structures it into TextPage instances.
  - PDF: Page-by-page extraction using PyMuPDF (fitz), preserving 1-based page numbers.
         Detects scanned/image-only PDFs and raises ScannedDocumentError (OCR not supported).
  - DOCX: Paragraph and table extraction using python-docx. Page numbers are None.
  - TXT: Multi-encoding decoding (UTF-8, UTF-8-sig, CP1252, Latin-1). Page numbers are None.
─────────────────────────────────────────────────────────────
"""

import logging
from pathlib import Path
from typing import List

from backend.schemas.chunk import TextPage

logger = logging.getLogger(__name__)

# Minimum characters across an entire multi-page PDF to qualify as having a text layer
SCANNED_PDF_MIN_TEXT_CHARS = 20


class ParsingError(Exception):
    """Base exception for parsing and extraction errors."""
    pass


class EmptyDocumentError(ParsingError):
    """Raised when the document file has 0 bytes or completely empty content."""
    pass


class DocumentParsingError(ParsingError):
    """Raised when a document cannot be opened or parsed due to corruption."""
    pass


class ScannedDocumentError(ParsingError):
    """Raised when a PDF contains scanned image pages without an extractable text layer."""
    pass


class UnsupportedFormatError(ParsingError):
    """Raised when an unsupported file type is provided."""
    pass


class DocumentParser:
    """Multi-format text extraction service."""

    @classmethod
    def parse_file(cls, file_path: Path, file_type: str) -> List[TextPage]:
        """
        Extract text pages/blocks from a document file.

        Args:
            file_path: Path to the target document.
            file_type: Extension without dot ('pdf', 'docx', 'txt').

        Returns:
            List of TextPage objects with text and page numbers.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        if file_path.stat().st_size == 0:
            raise EmptyDocumentError(f"Document file is 0 bytes: {file_path.name}")

        ft = file_type.lower().lstrip(".")
        if ft == "pdf":
            return cls.parse_pdf(file_path)
        elif ft == "docx":
            return cls.parse_docx(file_path)
        elif ft == "txt":
            return cls.parse_txt(file_path)
        else:
            raise UnsupportedFormatError(f"Unsupported document format: '{file_type}'")

    @classmethod
    def parse_pdf(cls, file_path: Path) -> List[TextPage]:
        """
        Extract text from a PDF file page by page using PyMuPDF.
        Preserves 1-based page numbers.
        Detects scanned PDFs with no text layer.
        """
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise DocumentParsingError(f"Failed to open PDF '{file_path.name}': {exc}") from exc

        try:
            total_pages = len(doc)
            if total_pages == 0:
                raise EmptyDocumentError(f"PDF file has 0 pages: {file_path.name}")

            pages: List[TextPage] = []
            total_extracted_chars = 0

            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                text = page.get_text("text") or ""
                total_extracted_chars += len(text.strip())
                pages.append(
                    TextPage(
                        page_number=page_idx + 1,  # 1-based index
                        text=text,
                        char_count=len(text),
                    )
                )

            # Check if document is an image-only / scanned PDF
            if total_extracted_chars < SCANNED_PDF_MIN_TEXT_CHARS:
                raise ScannedDocumentError(
                    f"PDF '{file_path.name}' contains no extractable text layer "
                    f"({total_extracted_chars} characters extracted across {total_pages} page(s)). "
                    "Scanned image PDFs have no extractable text (OCR not supported in Phase 4)."
                )

            return pages
        finally:
            doc.close()

    @classmethod
    def parse_docx(cls, file_path: Path) -> List[TextPage]:
        """
        Extract text from a DOCX file using python-docx.
        Extracts both paragraphs and table cells.
        Page numbers are explicitly None because DOCX is unpaginated flow content.
        """
        import docx
        from docx.opc.exceptions import PackageNotFoundError
        import zipfile

        try:
            doc = docx.Document(str(file_path))
        except (PackageNotFoundError, zipfile.BadZipFile) as exc:
            raise DocumentParsingError(f"Corrupted or invalid DOCX archive '{file_path.name}': {exc}") from exc
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse DOCX '{file_path.name}': {exc}") from exc

        parts: List[str] = []

        # 1. Extract paragraphs
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                parts.append(text)

        # 2. Extract tables as structured pipe-delimited text
        for table in doc.tables:
            table_lines: List[str] = []
            for row in table.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # Filter out empty rows
                if any(row_cells):
                    table_lines.append("| " + " | ".join(row_cells) + " |")
            if table_lines:
                parts.append("\n".join(table_lines))

        full_text = "\n\n".join(parts).strip()
        if not full_text:
            raise EmptyDocumentError(f"DOCX document '{file_path.name}' contains no extractable text.")

        # Page number is None for flow format
        return [TextPage(page_number=None, text=full_text, char_count=len(full_text))]

    @classmethod
    def parse_txt(cls, file_path: Path) -> List[TextPage]:
        """
        Extract text from a TXT file with fallback encodings.
        Page number is explicitly None.
        """
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) == 0:
            raise EmptyDocumentError(f"TXT file '{file_path.name}' is empty (0 bytes).")

        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        decoded_text: str | None = None

        for enc in encodings:
            try:
                decoded_text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise DocumentParsingError(
                f"Failed to decode TXT file '{file_path.name}' with supported encodings {encodings}"
            )

        stripped = decoded_text.strip()
        if not stripped:
            raise EmptyDocumentError(f"TXT file '{file_path.name}' contains only whitespace.")

        return [TextPage(page_number=None, text=stripped, char_count=len(stripped))]
