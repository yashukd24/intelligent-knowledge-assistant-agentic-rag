"""
tests/test_answer_verifier.py
───────────────────────────────────────────────────────────────────────────────
Unit and integration tests for Phase 15: Answer Synthesis, Verification and Citations.

Verifies:
  1. SynthesisVerification and SentenceGrounding schema validation.
  2. Grounded answer verification with strong lexical and citation alignment.
  3. Hallucination detection for fabricated or invalid citations ([S99]).
  4. Factual number inconsistency detection (fabricated financial or quantitative figures).
  5. Uncited substantive claims detection.
  6. AgentController end-to-end integration and verification state tracking.
───────────────────────────────────────────────────────────────────────────────
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from backend.schemas.agent_state import (
    AgentEventType,
    AgentRunRequest,
    AgentStatus,
)
from backend.schemas.rag import SourceReference
from backend.schemas.retrieval import RetrievedChunk
from backend.schemas.synthesis import SentenceGrounding, SynthesisVerification
from backend.services.agent_controller import AgentController
from backend.services.answer_generator import (
    AnswerGenerationResult,
    AnswerGenerator,
    AnswerStatus,
)
from backend.services.answer_verifier import AnswerVerifier
from backend.services.citation_validator import CitationValidationResult, CitationValidator
from backend.services.context_builder import ContextBuilder
from backend.services.prompt_builder import PromptBuilder
from backend.services.rag_service import RAGService
from backend.services.retrieval_service import RetrievalService


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_source_and_chunk(source_id: str, text: str, similarity_score: float = 0.88):
    doc_id = str(uuid4())
    col_id = str(uuid4())
    chunk_id = f"chunk-{source_id.replace('[', '').replace(']', '')}"
    
    src = SourceReference(
        source_id=source_id,
        document_id=doc_id,
        chunk_id=chunk_id,
        filename="report.pdf",
        page_number=1,
        chunk_index=0,
        similarity_score=similarity_score,
        char_count=len(text),
    )
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        collection_id=col_id,
        text=text,
        similarity_score=similarity_score,
        distance=round(1.0 - similarity_score, 4),
        chunk_index=0,
    )
    return src, chunk


# ── 1. Schema Validation Tests ───────────────────────────────────────────────


def test_synthesis_verification_schema():
    sg = SentenceGrounding(
        sentence="SQLite is a serverless database [S1].",
        cited_sources=["[S1]"],
        is_grounded=True,
        overlap_score=0.80,
    )
    sv = SynthesisVerification(
        is_verified=True,
        grounding_score=1.0,
        total_sentences=1,
        grounded_sentences=1,
        valid_citations=["[S1]"],
        sentence_evaluations=[sg],
        grounding_explanation="Fully grounded.",
    )
    assert sv.is_verified is True
    assert sv.grounding_score == 1.0
    assert sv.sentence_evaluations[0].sentence == "SQLite is a serverless database [S1]."


# ── 2. Answer Verifier Core Logic Tests ──────────────────────────────────────


def test_verifier_grounded_answer_success():
    src1, chk1 = make_source_and_chunk("[S1]", "Operating revenue in Q3 2025 reached $4.2B with solid margin expansion.")
    src2, chk2 = make_source_and_chunk("[S2]", "Capital expenditures were $600M focused on data center hardware.")

    verifier = AnswerVerifier()
    answer = "Operating revenue in Q3 2025 reached $4.2B [S1]. Capital expenditures were $600M [S2]."

    result = verifier.verify(answer_text=answer, sources=[src1, src2], chunks=[chk1, chk2])

    assert result.is_verified is True
    assert result.grounding_score == 1.0
    assert result.hallucination_warning is False
    assert result.valid_citations == ["[S1]", "[S2]"]
    assert len(result.invalid_citations) == 0
    assert result.grounded_sentences == 2


def test_verifier_detects_invalid_citation():
    src1, chk1 = make_source_and_chunk("[S1]", "Operating revenue was $4.2B.")

    verifier = AnswerVerifier()
    answer = "Operating revenue was $4.2B [S1]. Global expansion started in 2026 [S99]."

    result = verifier.verify(answer_text=answer, sources=[src1], chunks=[chk1])

    assert result.is_verified is False
    assert result.hallucination_warning is True
    assert "[S99]" in result.invalid_citations
    assert any("[S99]" in (sg.unsupported_reason or "") for sg in result.sentence_evaluations)


def test_verifier_detects_numeric_hallucination():
    # Source says $4.2B, but answer claims $9.9B
    src1, chk1 = make_source_and_chunk("[S1]", "Operating revenue in Q3 2025 reached $4.2B.")

    verifier = AnswerVerifier()
    answer = "Operating revenue in Q3 2025 was $9.9B [S1]."

    result = verifier.verify(answer_text=answer, sources=[src1], chunks=[chk1])

    assert result.is_verified is False
    assert result.hallucination_warning is True
    # The sentence with mismatched numbers must fail grounding
    eval_sg = result.sentence_evaluations[0]
    assert eval_sg.is_grounded is False
    assert "ungrounded numeric claim" in (eval_sg.unsupported_reason or "")


def test_verifier_detects_uncited_substantive_claim():
    src1, chk1 = make_source_and_chunk("[S1]", "Operating revenue in Q3 2025 was $4.2B.")

    verifier = AnswerVerifier()
    answer = (
        "Operating revenue in Q3 2025 was $4.2B [S1]. "
        "The company also launched three separate quantum computing satellites into orbit."
    )

    result = verifier.verify(answer_text=answer, sources=[src1], chunks=[chk1])

    assert result.is_verified is False
    assert result.total_sentences == 2
    assert result.grounded_sentences == 1
    # Second sentence lacked citation
    assert result.sentence_evaluations[1].is_grounded is False
    assert "lacks source citation" in (result.sentence_evaluations[1].unsupported_reason or "")


# ── 3. Controller Integration with Verification ──────────────────────────────


@pytest.mark.asyncio
async def test_agent_controller_end_to_end_verification():
    col_id = str(uuid4())
    doc_id = str(uuid4())

    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id=doc_id,
        collection_id=col_id,
        text="The Intelligent Knowledge Assistant uses a plain Python state machine without LangChain.",
        similarity_score=0.93,
        distance=0.07,
        chunk_index=0,
    )

    mock_resp = MagicMock()
    mock_resp.chunks = [chunk]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=mock_resp)

    rag_svc = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="The Intelligent Knowledge Assistant uses a plain Python state machine [S1].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet",
            latency_ms=260.0,
            citations=CitationValidationResult(
                cited_sources=["[S1]"],
                valid_citations=["[S1]"],
            ),
        )
    )

    controller = AgentController(
        rag_service=rag_svc,
        answer_generator=mock_answer_gen,
    )

    req = AgentRunRequest(query="What state machine is used?")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.verification is not None
    assert resp.verification.is_verified is True
    assert resp.verification.grounding_score >= 0.80
    assert resp.verification.hallucination_warning is False

    # Verify ANSWER_VERIFIED event recorded in event stream
    verified_events = [e for e in resp.events if e.event_type == AgentEventType.ANSWER_VERIFIED]
    assert len(verified_events) == 1
    assert verified_events[0].payload["is_verified"] is True
