"""
tests/test_citation_validator.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for CitationValidator and source tag verification.
Verifies citation parsing, hallucinated citation detection, uncited sources,
and citation metadata integrity.
─────────────────────────────────────────────────────────────
"""

from uuid import uuid4
import pytest

from backend.schemas.rag import SourceReference
from backend.services.citation_validator import CitationValidator


def _create_source(source_num: int, filename: str = "doc.pdf", page: int | None = 1) -> SourceReference:
    return SourceReference(
        source_id=f"[S{source_num}]",
        document_id=str(uuid4()),
        chunk_id=str(uuid4()),
        filename=filename,
        page_number=page,
        chunk_index=source_num - 1,
        similarity_score=0.85,
        char_count=120,
    )


def test_validate_citations_single():
    validator = CitationValidator()
    sources = [_create_source(1)]
    text = "The system requirements are detailed in [S1]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.cited_sources == ["[S1]"]
    assert result.valid_citations == ["[S1]"]
    assert result.invalid_citations == []
    assert result.uncited_sources == []
    assert len(result.valid_source_references) == 1
    assert result.valid_source_references[0].source_id == "[S1]"


def test_validate_citations_multiple_separate():
    validator = CitationValidator()
    sources = [_create_source(1), _create_source(2)]
    text = "Requirement A is in [S1], and requirement B is in [S2]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.cited_sources == ["[S1]", "[S2]"]
    assert result.valid_citations == ["[S1]", "[S2]"]
    assert result.invalid_citations == []
    assert result.uncited_sources == []


def test_validate_citations_adjacent():
    validator = CitationValidator()
    sources = [_create_source(1), _create_source(2)]
    text = "Both criteria are discussed extensively [S1][S2]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.cited_sources == ["[S1]", "[S2]"]
    assert result.valid_citations == ["[S1]", "[S2]"]
    assert result.invalid_citations == []
    assert result.uncited_sources == []


def test_validate_citations_duplicates():
    validator = CitationValidator()
    sources = [_create_source(1), _create_source(2)]
    text = "As stated in [S1], we must verify [S2]. Furthermore, [S1] confirms this."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.cited_sources == ["[S1]", "[S2]"]
    assert result.valid_citations == ["[S1]", "[S2]"]
    assert result.invalid_citations == []
    assert result.uncited_sources == []


def test_validate_citations_none_cited():
    validator = CitationValidator()
    sources = [_create_source(1)]
    text = "There are no references cited in this text."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is False
    assert result.cited_sources == []
    assert result.valid_citations == []
    assert result.invalid_citations == []
    assert result.uncited_sources == ["[S1]"]


def test_validate_citations_malformed_ignored():
    validator = CitationValidator()
    sources = [_create_source(1)]
    text = "Invalid citations: [S], [Sabc], [1], [Source 1], (S1), S1, [s1]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is False
    assert result.cited_sources == []
    assert result.uncited_sources == ["[S1]"]


def test_validate_citations_empty_string():
    validator = CitationValidator()
    result = validator.validate_citations("", [_create_source(1)])
    assert result.has_citations is False
    assert result.cited_sources == []


def test_validate_citations_with_uncited_sources():
    validator = CitationValidator()
    sources = [_create_source(1), _create_source(2), _create_source(3)]
    text = "Only the first source is cited [S1]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.valid_citations == ["[S1]"]
    assert result.invalid_citations == []
    assert result.uncited_sources == ["[S2]", "[S3]"]
    assert len(result.valid_source_references) == 1


def test_validate_citations_with_hallucinated_tags():
    validator = CitationValidator()
    sources = [_create_source(1), _create_source(2)]
    text = "Fact from valid source [S1], but also from phantom source [S99]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.has_invalid_citations is True
    assert result.valid_citations == ["[S1]"]
    assert result.invalid_citations == ["[S99]"]
    assert result.uncited_sources == ["[S2]"]
    assert len(result.valid_source_references) == 1
    assert result.valid_source_references[0].source_id == "[S1]"


def test_validate_citations_all_hallucinated():
    validator = CitationValidator()
    sources = [_create_source(1)]
    text = "Everything cited here is completely hallucinated [S42][S43]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.has_invalid_citations is True
    assert result.valid_citations == []
    assert result.invalid_citations == ["[S42]", "[S43]"]
    assert result.uncited_sources == ["[S1]"]
    assert len(result.valid_source_references) == 0


def test_validate_citations_empty_sources_and_no_citations():
    validator = CitationValidator()
    sources = []
    text = "I do not have sufficient information in the provided documents to answer this question."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is False
    assert result.has_invalid_citations is False
    assert result.valid_citations == []
    assert result.invalid_citations == []
    assert result.uncited_sources == []
    assert len(result.valid_source_references) == 0


def test_validate_citations_empty_sources_with_hallucinated_citations():
    validator = CitationValidator()
    sources = []
    text = "Here is an answer anyway [S1]."

    result = validator.validate_citations(text, sources)

    assert result.has_citations is True
    assert result.has_invalid_citations is True
    assert result.valid_citations == []
    assert result.invalid_citations == ["[S1]"]
    assert result.uncited_sources == []
