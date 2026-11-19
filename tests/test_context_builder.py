"""
tests/test_context_builder.py
─────────────────────────────────────────────────────────────
Comprehensive tests for ContextBuilder:
  - Single and multiple retrieved chunk formatting.
  - Deterministic source ID assignment ([S1], [S2], ...).
  - Preservation of retrieval ranking order and exact source text.
  - Filename, document, and page metadata preservation.
  - Empty chunk lists and no-evidence handling.
  - Context character budget enforcement and whole-chunk exclusion.
  - Duplicate chunk rejection.
  - Cross-collection chunk isolation.
  - Source-to-chunk mapping correctness.
─────────────────────────────────────────────────────────────
"""

from uuid import uuid4

import pytest

from backend.schemas.rag import ExcludedChunkInfo, SourceReference
from backend.schemas.retrieval import RetrievedChunk
from backend.services.context_builder import ContextBuilder, ContextBuildResult

TEST_COL_ID = str(uuid4())
OTHER_COL_ID = str(uuid4())


def make_chunk(
    chunk_id: str = "c_1",
    document_id: str = "doc_1",
    collection_id: str = TEST_COL_ID,
    text: str = "Sample chunk content for testing.",
    similarity_score: float = 0.85,
    distance: float = 0.15,
    chunk_index: int = 0,
    filename: str = "policy.pdf",
    page_number: int | None = 1,
    page_range: str | None = None,
) -> RetrievedChunk:
    eff_range = page_range if page_range is not None else (str(page_number) if page_number is not None else None)
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection_id=collection_id,
        text=text,
        similarity_score=similarity_score,
        distance=distance,
        chunk_index=chunk_index,
        filename=filename,
        page_number=page_number,
        page_range=eff_range,
    )


def test_empty_chunks_returns_empty_result():
    builder = ContextBuilder()
    res = builder.build_context([])
    assert isinstance(res, ContextBuildResult)
    assert res.assembled_context == ""
    assert res.sources == []
    assert res.excluded_chunks == []
    assert res.context_character_count == 0
    assert res.source_metadata == {}
    assert res.has_evidence is False


def test_single_retrieved_chunk_formatting():
    builder = ContextBuilder()
    chunk = make_chunk(
        chunk_id="chk_01",
        document_id="doc_01",
        text="Academic integrity is vital for university students.",
        filename="handbook.pdf",
        page_number=3,
        similarity_score=0.9123,
    )
    res = builder.build_context([chunk])
    assert res.has_evidence is True
    assert len(res.sources) == 1
    assert res.sources[0].source_id == "[S1]"
    assert res.sources[0].chunk_id == "chk_01"
    assert res.sources[0].document_id == "doc_01"
    assert res.sources[0].filename == "handbook.pdf"
    assert res.sources[0].page_number == 3
    assert res.sources[0].similarity_score == 0.9123

    # Verify context block contains source header and text
    assert "[S1] Source: handbook.pdf (Doc ID: doc_01, Chunk ID: chk_01, Page 3)" in res.assembled_context
    assert "Similarity Score: 0.9123 (retrieval metric; not proof of factual correctness)" in res.assembled_context
    assert "Academic integrity is vital for university students." in res.assembled_context
    assert res.context_character_count == len(res.assembled_context)
    assert "[S1]" in res.source_metadata


def test_multiple_chunks_preserves_ranking_and_source_ids():
    builder = ContextBuilder()
    c1 = make_chunk(chunk_id="c1", text="First chunk text.", similarity_score=0.95, filename="a.pdf")
    c2 = make_chunk(chunk_id="c2", text="Second chunk text.", similarity_score=0.85, filename="b.pdf")
    c3 = make_chunk(chunk_id="c3", text="Third chunk text.", similarity_score=0.75, filename="c.pdf")

    res = builder.build_context([c1, c2, c3])
    assert len(res.sources) == 3
    assert [s.source_id for s in res.sources] == ["[S1]", "[S2]", "[S3]"]
    assert [s.chunk_id for s in res.sources] == ["c1", "c2", "c3"]
    assert len(res.excluded_chunks) == 0

    # Ensure ranking order is preserved in context string
    pos_s1 = res.assembled_context.find("[S1]")
    pos_s2 = res.assembled_context.find("[S2]")
    pos_s3 = res.assembled_context.find("[S3]")
    assert 0 <= pos_s1 < pos_s2 < pos_s3


def test_page_number_and_page_range_formatting():
    builder = ContextBuilder()
    # Case A: page_number and page_range same
    cA = make_chunk(chunk_id="cA", page_number=5, page_range="5")
    # Case B: page_range span
    cB = make_chunk(chunk_id="cB", page_number=5, page_range="5-7")
    # Case C: no page info
    cC = make_chunk(chunk_id="cC", page_number=None, page_range=None)

    res = builder.build_context([cA, cB, cC])
    assert "Page 5" in res.assembled_context
    assert "Pages 5-7" in res.assembled_context
    assert "Page N/A" in res.assembled_context


def test_duplicate_chunk_rejection():
    builder = ContextBuilder()
    c1 = make_chunk(chunk_id="c_dup", text="First occurrence.")
    c2 = make_chunk(chunk_id="c_dup", text="Duplicate occurrence.")
    c3 = make_chunk(chunk_id="c_other", text="Unique chunk.")

    res = builder.build_context([c1, c2, c3])
    assert len(res.sources) == 2
    assert [s.source_id for s in res.sources] == ["[S1]", "[S2]"]
    assert res.sources[0].chunk_id == "c_dup"
    assert res.sources[1].chunk_id == "c_other"

    assert len(res.excluded_chunks) == 1
    assert res.excluded_chunks[0].chunk_id == "c_dup"
    assert res.excluded_chunks[0].reason == "duplicate_chunk_id"


def test_cross_collection_chunk_isolation():
    builder = ContextBuilder()
    c_valid = make_chunk(chunk_id="c_val", collection_id=TEST_COL_ID)
    c_cross = make_chunk(chunk_id="c_cro", collection_id=OTHER_COL_ID)

    res = builder.build_context([c_valid, c_cross], expected_collection_id=TEST_COL_ID)
    assert len(res.sources) == 1
    assert res.sources[0].chunk_id == "c_val"

    assert len(res.excluded_chunks) == 1
    assert res.excluded_chunks[0].chunk_id == "c_cro"
    assert res.excluded_chunks[0].reason == "cross_collection_mismatch"


def test_context_budget_whole_chunk_exclusion():
    builder = ContextBuilder()
    # A single chunk with header is roughly ~150 chars
    long_text = "A" * 300
    c1 = make_chunk(chunk_id="c1", text=long_text)
    c2 = make_chunk(chunk_id="c2", text=long_text)
    c3 = make_chunk(chunk_id="c3", text=long_text)

    # Allow enough for c1 + header (~500 chars), but not c2
    res = builder.build_context([c1, c2, c3], max_context_chars=600)
    assert len(res.sources) == 1
    assert res.sources[0].chunk_id == "c1"
    assert len(res.excluded_chunks) == 2
    assert res.excluded_chunks[0].chunk_id == "c2"
    assert res.excluded_chunks[0].reason == "budget_exceeded"
    assert res.excluded_chunks[1].chunk_id == "c3"
    assert res.excluded_chunks[1].reason == "budget_exceeded"
    assert res.context_character_count <= 600


def test_invalid_budget_bounds():
    builder = ContextBuilder()
    with pytest.raises(ValueError, match="max_context_chars"):
        builder.build_context([], max_context_chars=100)  # Below 500

    with pytest.raises(ValueError, match="max_context_chars"):
        builder.build_context([], max_context_chars=100000)  # Above 60000


def test_source_to_chunk_mapping_exactness():
    builder = ContextBuilder()
    c1 = make_chunk(chunk_id="chunk_alpha", document_id="doc_alpha", similarity_score=0.8888)
    c2 = make_chunk(chunk_id="chunk_beta", document_id="doc_beta", similarity_score=0.7777)

    res = builder.build_context([c1, c2])
    meta_s1 = res.source_metadata["[S1]"]
    assert meta_s1["chunk_id"] == "chunk_alpha"
    assert meta_s1["document_id"] == "doc_alpha"
    assert meta_s1["similarity_score"] == 0.8888

    meta_s2 = res.source_metadata["[S2]"]
    assert meta_s2["chunk_id"] == "chunk_beta"
    assert meta_s2["document_id"] == "doc_beta"
    assert meta_s2["similarity_score"] == 0.7777
