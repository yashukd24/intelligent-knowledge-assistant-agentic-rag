"""
tests/test_evidence_evaluator.py
───────────────────────────────────────────────────────────────────────────────
Unit and integration tests for Phase 13: Evidence Evaluation and Sufficiency Decisions.

Verifies:
  1. Typed schema invariants and validation.
  2. Empty retrieval gating (EMPTY_RETRIEVAL).
  3. Context too brief gating (CONTEXT_TOO_BRIEF).
  4. Low vector similarity score gating (LOW_SIMILARITY_SCORE).
  5. Subquery coverage calculation and gating (INSUFFICIENT_SUBQUERY_COVERAGE).
  6. Keyword overlap calculation and composite scoring.
  7. Successful multi-criteria qualification (SUFFICIENT_EVIDENCE).
  8. AgentController end-to-end integration and state tracking.
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
from backend.schemas.evidence import (
    ChunkRelevance,
    EvidenceEvaluation,
    SubqueryCoverage,
    SufficiencyReason,
)
from backend.schemas.query_plan import QueryIntent, SubQuery
from backend.schemas.retrieval import RetrievedChunk
from backend.services.agent_controller import AgentController
from backend.services.answer_generator import (
    AnswerGenerationResult,
    AnswerGenerator,
    AnswerStatus,
    STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
)
from backend.services.citation_validator import CitationValidationResult
from backend.services.context_builder import ContextBuilder
from backend.services.evidence_evaluator import EvidenceEvaluator
from backend.services.prompt_builder import PromptBuilder
from backend.services.rag_service import RAGService
from backend.services.retrieval_service import RetrievalService


# ── Fixtures & Helpers ───────────────────────────────────────────────────────


def make_chunk(
    chunk_id: str,
    text: str,
    similarity_score: float = 0.85,
    doc_id: str = "doc-1",
    col_id: str = "col-1",
    index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        collection_id=col_id,
        text=text,
        similarity_score=similarity_score,
        distance=round(1.0 - similarity_score, 4),
        chunk_index=index,
    )


# ── 1. Schema Validation Tests ───────────────────────────────────────────────


def test_chunk_relevance_schema():
    cr = ChunkRelevance(
        chunk_id="c-1",
        similarity_score=0.88,
        keyword_overlap_ratio=0.5,
        composite_score=0.785,
        passes_threshold=True,
    )
    assert cr.chunk_id == "c-1"
    assert cr.passes_threshold is True
    assert cr.composite_score == 0.785


def test_subquery_coverage_schema():
    sc = SubqueryCoverage(
        subquery_id="sq-1",
        subquery_text="What was Q3 revenue?",
        matched_chunk_ids=["c-1"],
        highest_similarity=0.91,
        is_covered=True,
    )
    assert sc.subquery_id == "sq-1"
    assert sc.is_covered is True
    assert len(sc.matched_chunk_ids) == 1


def test_evidence_evaluation_schema():
    eval_rep = EvidenceEvaluation(
        overall_score=0.85,
        is_sufficient=True,
        reason=SufficiencyReason.SUFFICIENT_EVIDENCE,
        explanation="Adequate evidence found.",
        chunk_evaluations=[],
        subquery_coverages=[],
        coverage_ratio=1.0,
        min_similarity_threshold=0.50,
        total_chunks_evaluated=2,
        qualifying_chunk_count=2,
    )
    assert eval_rep.is_sufficient is True
    assert eval_rep.reason == SufficiencyReason.SUFFICIENT_EVIDENCE


# ── 2. Evidence Evaluator Rule Gating Tests ──────────────────────────────────


def test_evaluator_empty_retrieval():
    evaluator = EvidenceEvaluator(default_similarity_threshold=0.50)
    res = evaluator.evaluate(
        query="What was the net profit in 2024?",
        chunks=[],
        assembled_context="",
    )
    assert res.is_sufficient is False
    assert res.reason == SufficiencyReason.EMPTY_RETRIEVAL
    assert res.overall_score == 0.0
    assert res.total_chunks_evaluated == 0
    assert "No document chunks were retrieved" in res.explanation


def test_evaluator_context_too_brief():
    evaluator = EvidenceEvaluator(min_context_chars=30)
    chunk = make_chunk("c-1", "Too short", similarity_score=0.90)
    res = evaluator.evaluate(
        query="Explain system architecture",
        chunks=[chunk],
        assembled_context="Too short",
    )
    assert res.is_sufficient is False
    assert res.reason == SufficiencyReason.CONTEXT_TOO_BRIEF
    assert "below the minimum threshold" in res.explanation


def test_evaluator_low_similarity_score():
    evaluator = EvidenceEvaluator(default_similarity_threshold=0.60)
    chunk = make_chunk(
        "c-1",
        "The quick brown fox jumps over the lazy dog in the park during summer.",
        similarity_score=0.42,
    )
    res = evaluator.evaluate(
        query="Quantum computing encryption algorithms",
        chunks=[chunk],
        assembled_context="The quick brown fox jumps over the lazy dog in the park during summer.",
    )
    assert res.is_sufficient is False
    assert res.reason == SufficiencyReason.LOW_SIMILARITY_SCORE
    assert "below the required threshold" in res.explanation


def test_evaluator_insufficient_subquery_coverage():
    evaluator = EvidenceEvaluator(default_similarity_threshold=0.50, min_coverage_ratio=0.75)
    # 2 subqueries: one covered, one completely uncovered
    parent = "What was the operating revenue in Q3 2025 and the Tokyo litigation settlement?"
    sq1 = SubQuery(id="sq-1", text="What was the operating revenue in Q3 2025?", intent=QueryIntent.FACTUAL, parent_query=parent)
    sq2 = SubQuery(id="sq-2", text="What was the litigation settlement in Tokyo?", intent=QueryIntent.FACTUAL, parent_query=parent)

    chunk = make_chunk(
        "c-1",
        "Operating revenue in Q3 2025 reached $4.2B with steady margin expansion.",
        similarity_score=0.89,
    )

    res = evaluator.evaluate(
        query="What was the operating revenue in Q3 2025 and the Tokyo litigation settlement?",
        chunks=[chunk],
        subqueries=[sq1, sq2],
        assembled_context="Operating revenue in Q3 2025 reached $4.2B with steady margin expansion.",
    )

    assert res.is_sufficient is False
    assert res.reason == SufficiencyReason.INSUFFICIENT_SUBQUERY_COVERAGE
    assert res.coverage_ratio == 0.50
    assert len(res.subquery_coverages) == 2
    assert res.subquery_coverages[0].is_covered is True
    assert res.subquery_coverages[1].is_covered is False


def test_evaluator_sufficient_multi_part_evidence():
    evaluator = EvidenceEvaluator(default_similarity_threshold=0.50, min_coverage_ratio=0.50)
    parent = "What was the operating revenue and CapEx in Q3 2025?"
    sq1 = SubQuery(id="sq-1", text="What was the operating revenue in Q3 2025?", intent=QueryIntent.FACTUAL, parent_query=parent)
    sq2 = SubQuery(id="sq-2", text="What were capital expenditures in Q3 2025?", intent=QueryIntent.FACTUAL, parent_query=parent)

    chunk1 = make_chunk(
        "c-1",
        "Operating revenue in Q3 2025 reached $4.2B with strong cloud growth.",
        similarity_score=0.88,
    )
    chunk2 = make_chunk(
        "c-2",
        "Capital expenditures in Q3 2025 were $600M focused on infrastructure.",
        similarity_score=0.85,
    )

    context = (
        "Operating revenue in Q3 2025 reached $4.2B with strong cloud growth. "
        "Capital expenditures in Q3 2025 were $600M focused on infrastructure."
    )

    res = evaluator.evaluate(
        query="What was the operating revenue and CapEx in Q3 2025?",
        chunks=[chunk1, chunk2],
        subqueries=[sq1, sq2],
        assembled_context=context,
    )

    assert res.is_sufficient is True
    assert res.reason == SufficiencyReason.SUFFICIENT_EVIDENCE
    assert res.coverage_ratio == 1.0
    assert res.qualifying_chunk_count == 2
    assert res.overall_score >= 0.80


# ── 3. Controller Integration with Evidence Evaluator ────────────────────────


@pytest.mark.asyncio
async def test_agent_controller_sufficient_evidence_pipeline():
    col_id = str(uuid4())
    doc_id = str(uuid4())

    chunk = make_chunk(
        "chunk-1",
        "SQLite is a C-language library that implements a small, fast, self-contained SQL database engine.",
        similarity_score=0.91,
        doc_id=doc_id,
        col_id=col_id,
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
            answer="SQLite is a fast, self-contained SQL database engine [S1].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet",
            latency_ms=250.0,
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

    req = AgentRunRequest(query="What is SQLite?")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is True
    assert resp.evidence_evaluation is not None
    assert resp.evidence_evaluation.is_sufficient is True
    assert resp.evidence_evaluation.reason == SufficiencyReason.SUFFICIENT_EVIDENCE
    assert resp.evidence_evaluation.qualifying_chunk_count == 1
    assert "SQLite is a fast" in resp.answer


@pytest.mark.asyncio
async def test_agent_controller_insufficient_evidence_safely_refuses():
    col_id = str(uuid4())

    # Empty retrieval response
    mock_resp = MagicMock()
    mock_resp.chunks = []

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=mock_resp)

    rag_svc = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)

    controller = AgentController(
        rag_service=rag_svc,
        answer_generator=mock_answer_gen,
    )

    req = AgentRunRequest(query="Unknown topic without documents")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is False
    assert resp.evidence_evaluation is not None
    assert resp.evidence_evaluation.is_sufficient is False
    assert resp.evidence_evaluation.reason == SufficiencyReason.EMPTY_RETRIEVAL
    assert resp.answer == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
    # LLM inference must NOT be called on insufficient evidence
    mock_answer_gen.generate_answer.assert_not_called()
