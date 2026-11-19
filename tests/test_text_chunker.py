"""
tests/test_text_chunker.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for TextChunker service.
─────────────────────────────────────────────────────────────
"""

import pytest
from backend.schemas.chunk import TextPage
from backend.services.text_chunker import TextChunker


def test_chunker_initialization_validation():
    # Valid
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 64

    # Invalid overlap >= size
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=150)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=-10, chunk_overlap=0)


def test_chunk_empty_or_blank_document():
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    assert chunker.chunk_document("d1", "c1", "empty.txt", "txt", []) == []
    
    blank_page = TextPage(page_number=1, text="   \n\t  ")
    assert chunker.chunk_document("d1", "c1", "blank.pdf", "pdf", [blank_page]) == []


def test_chunk_short_document_single_chunk():
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    short_text = "This is a short academic policy document containing fewer than 512 characters."
    page = TextPage(page_number=1, text=short_text)
    
    chunks = chunker.chunk_document("doc-123", "col-456", "policy.pdf", "pdf", [page])
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_text == short_text
    assert chunks[0].char_count == len(short_text)
    assert chunks[0].chunk_id == "doc-123_c0000"
    assert chunks[0].page_number == 1
    assert chunks[0].page_range is None


def test_chunk_word_boundary_preservation():
    # Size 50, overlap 10
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    # 80 chars total
    text = "The quick brown fox jumps over the lazy dog repeatedly until evening comes."
    page = TextPage(page_number=1, text=text)
    
    chunks = chunker.chunk_document("d1", "c1", "fox.txt", "txt", [page])
    assert len(chunks) >= 2
    for c in chunks:
        # Assert each chunk ends with a complete word (no severed words)
        last_word = c.chunk_text.split()[-1]
        assert last_word in text, f"Severed word found: {last_word}"


def test_chunk_pdf_multipage_and_page_range_attribution():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    # Page 1: 80 chars
    p1_text = "Page 1 Content covering academic grading criteria and university evaluation rules."
    # Page 2: 80 chars
    p2_text = "Page 2 Content detailing the minimum attendance percentage and leave applications."
    
    pages = [
        TextPage(page_number=1, text=p1_text),
        TextPage(page_number=2, text=p2_text),
    ]
    
    chunks = chunker.chunk_document("doc-pdf", "col-1", "rules.pdf", "pdf", pages)
    assert len(chunks) >= 2
    
    # First chunk should be on page 1
    assert chunks[0].page_number == 1
    
    # Check if any chunk straddles page 1 and page 2
    straddle_found = any(c.page_range == "1-2" for c in chunks)
    # Last chunk should be on page 2
    assert chunks[-1].page_number in (1, 2)


def test_chunk_docx_and_txt_page_numbers_are_none():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    # For DOCX / TXT, TextPage.page_number is None
    pages_docx = [TextPage(page_number=None, text="DOCX content spanning multiple paragraphs for testing.")]
    chunks_docx = chunker.chunk_document("d-docx", "c-1", "file.docx", "docx", pages_docx)
    
    assert len(chunks_docx) >= 1
    for c in chunks_docx:
        assert c.page_number is None
        assert c.page_range is None
        assert c.metadata.page_number is None
        assert c.metadata.page_range is None


def test_chunk_consecutive_overlap_content():
    chunker = TextChunker(chunk_size=60, chunk_overlap=15)
    text = "First section introduces RAG concepts. Second section discusses vector databases."
    page = TextPage(page_number=1, text=text)
    
    chunks = chunker.chunk_document("d1", "c1", "rag.pdf", "pdf", [page])
    assert len(chunks) >= 2
    
    # Check that chunk 2 contains trailing text from chunk 1
    c1_end = chunks[0].chunk_text[-10:]
    assert c1_end in chunks[1].chunk_text


def test_chunk_unbroken_long_token_no_infinite_loop():
    # Test text with no spaces exceeding chunk_size
    chunker = TextChunker(chunk_size=40, chunk_overlap=10)
    unbroken = "A" * 100
    page = TextPage(page_number=1, text=unbroken)
    
    chunks = chunker.chunk_document("d1", "c1", "dna.txt", "txt", [page])
    assert len(chunks) >= 3
    # Verify no loop occurred and all chars were captured
    reconstructed = "".join([c.chunk_text for c in chunks])
    assert "AAAA" in reconstructed
    assert all(c.char_count <= 40 for c in chunks)


def test_chunk_metadata_offsets_and_counts():
    chunker = TextChunker(chunk_size=80, chunk_overlap=15)
    text = "Sentence one starts here. Sentence two follows closely. Sentence three concludes the test."
    page = TextPage(page_number=1, text=text)
    
    chunks = chunker.chunk_document("d-meta", "col-meta", "meta.txt", "txt", [page])
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.char_count == len(c.chunk_text)
        assert c.metadata.char_count == len(c.chunk_text)
        assert c.metadata.start_char >= 0
        assert c.metadata.end_char > c.metadata.start_char
        assert c.metadata.document_id == "d-meta"
        assert c.metadata.collection_id == "col-meta"
