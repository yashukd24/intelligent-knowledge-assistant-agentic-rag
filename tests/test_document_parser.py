"""
tests/test_document_parser.py
─────────────────────────────────────────────────────────────
Comprehensive tests for DocumentParser service (PDF, DOCX, TXT).
─────────────────────────────────────────────────────────────
"""

from pathlib import Path
import pytest
import pymupdf as fitz
import docx

from backend.services.document_parser import (
    DocumentParser,
    EmptyDocumentError,
    DocumentParsingError,
    ScannedDocumentError,
    UnsupportedFormatError,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Generate a valid 2-page PDF file."""
    pdf_path = tmp_path / "academic_regulations.pdf"
    doc = fitz.open()
    
    # Page 1
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Dayananda Sagar University Regulations 2026. Page 1 Content.")
    
    # Page 2
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Section 2: Examination and Grading Rules. Page 2 Content.")
    
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """Generate a PDF with no extractable text (simulating scanned/image PDF)."""
    pdf_path = tmp_path / "scanned_doc.pdf"
    doc = fitz.open()
    doc.new_page()  # Blank page with zero text
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    """Generate a valid DOCX with paragraphs and a table."""
    docx_path = tmp_path / "syllabus.docx"
    doc = docx.Document()
    doc.add_heading("Course Syllabus 2026", level=1)
    doc.add_paragraph("This course covers Agentic RAG and Large Language Models.")
    
    # Add a table
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Module"
    table.rows[0].cells[1].text = "Topic"
    table.rows[1].cells[0].text = "Unit 1"
    table.rows[1].cells[1].text = "Vector Databases and Embeddings"
    
    doc.save(str(docx_path))
    return docx_path


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """Generate a valid UTF-8 TXT document."""
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Intelligent Knowledge Assistant TXT notes.\nAll systems operational.", encoding="utf-8")
    return txt_path


# ── PDF Tests ──────────────────────────────────────────────

def test_parse_pdf_multipage_success(sample_pdf: Path):
    pages = DocumentParser.parse_file(sample_pdf, "pdf")
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Page 1 Content" in pages[0].text
    assert pages[1].page_number == 2
    assert "Page 2 Content" in pages[1].text


def test_parse_pdf_scanned_rejected(scanned_pdf: Path):
    with pytest.raises(ScannedDocumentError) as exc_info:
        DocumentParser.parse_file(scanned_pdf, "pdf")
    assert "no extractable text layer" in str(exc_info.value).lower()
    assert "ocr not supported" in str(exc_info.value).lower()


def test_parse_pdf_corrupted(tmp_path: Path):
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\nCorrupted binary content without xref table")
    with pytest.raises(DocumentParsingError):
        DocumentParser.parse_file(corrupt_pdf, "pdf")


# ── DOCX Tests ─────────────────────────────────────────────

def test_parse_docx_paragraphs_and_tables(sample_docx: Path):
    pages = DocumentParser.parse_file(sample_docx, "docx")
    assert len(pages) == 1
    # Page number must be None for DOCX
    assert pages[0].page_number is None
    text = pages[0].text
    assert "Course Syllabus 2026" in text
    assert "Agentic RAG and Large Language Models" in text
    # Table extracted in pipe-delimited format
    assert "| Module | Topic |" in text
    assert "| Unit 1 | Vector Databases and Embeddings |" in text


def test_parse_docx_corrupted(tmp_path: Path):
    corrupt_docx = tmp_path / "corrupt.docx"
    corrupt_docx.write_bytes(b"PK\x03\x04NotAValidZipArchiveContent")
    with pytest.raises(DocumentParsingError):
        DocumentParser.parse_file(corrupt_docx, "docx")


# ── TXT Tests ──────────────────────────────────────────────

def test_parse_txt_utf8(sample_txt: Path):
    pages = DocumentParser.parse_file(sample_txt, "txt")
    assert len(pages) == 1
    assert pages[0].page_number is None
    assert "Intelligent Knowledge Assistant TXT notes" in pages[0].text


def test_parse_txt_latin1_fallback(tmp_path: Path):
    latin1_path = tmp_path / "latin1.txt"
    # Write text encoded in Latin-1 (contains byte that is invalid in standard UTF-8)
    latin1_text = "Regulations: Café cost is £5 per semester."
    latin1_path.write_bytes(latin1_text.encode("latin-1"))
    
    pages = DocumentParser.parse_file(latin1_path, "txt")
    assert len(pages) == 1
    assert "Café" in pages[0].text
    assert "£5" in pages[0].text


# ── Common Error Handling Tests ────────────────────────────

def test_parse_empty_files_rejected(tmp_path: Path):
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        DocumentParser.parse_file(empty_pdf, "pdf")

    empty_docx = tmp_path / "empty.docx"
    empty_docx.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        DocumentParser.parse_file(empty_docx, "docx")

    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        DocumentParser.parse_file(empty_txt, "txt")


def test_parse_unsupported_format(tmp_path: Path):
    excel_path = tmp_path / "data.xlsx"
    excel_path.write_bytes(b"dummy content")
    with pytest.raises(UnsupportedFormatError):
        DocumentParser.parse_file(excel_path, "xlsx")


def test_parse_missing_file():
    missing = Path("data/uploads/missing_collection/missing_file.pdf")
    with pytest.raises(FileNotFoundError):
        DocumentParser.parse_file(missing, "pdf")
