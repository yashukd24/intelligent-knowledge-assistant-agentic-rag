"""
tests/test_query_refiner.py
───────────────────────────────────────────────────────────────────────────────
Unit and integration tests for Phase 14: Query Refinement and Adaptive Re-retrieval.

Verifies:
  1. RefinedQuery schema validation and immutability.
  2. can_refine termination guards.
  3. Refinement strategies (targeting uncovered subqueries, relaxing thresholds, broadening scope).
  4. AgentController integration with successful adaptive re-retrieval.
  5. AgentController bounded loop execution and refusal after max iterations.
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
from backend.schemas.query_refinement import RefinedQuery, RefinementStrategy
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
from backend.services.prompt_builder import PromptBuilder
from backend.services.query_refiner import QueryRefiner
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


def test_refined_query_schema():
    rq = RefinedQuery(
        iteration=1,
        strategy=RefinementStrategy.TARGET_UNCOVERED_SUBQUERIES,
        original_query="What was revenue and CapEx?",
        refined_query_text="What was CapEx?",
        targeted_subquery_ids=["sq-2"],
        adjusted_top_k=8,
        adjusted_similarity_threshold=0.40,
        rationale="Targeting missing CapEx data.",
    )
    assert rq.iteration == 1
    assert rq.strategy == RefinementStrategy.TARGET_UNCOVERED_SUBQUERIES
    assert rq.adjusted_top_k == 8
    assert rq.adjusted_similarity_threshold == 0.40


# ── 2. Query Refiner Unit Tests ──────────────────────────────────────────────


def test_can_refine_guards():
    refiner = QueryRefiner(max_refinements=2)

    eval_insufficient = EvidenceEvaluation(
        overall_score=0.30,
        is_sufficient=False,
        reason=SufficiencyReason.LOW_SIMILARITY_SCORE,
        explanation="Low scores",
        coverage_ratio=0.0,
    )
    eval_sufficient = EvidenceEvaluation(
        overall_score=0.90,
        is_sufficient=True,
        reason=SufficiencyReason.SUFFICIENT_EVIDENCE,
        explanation="Good",
        coverage_ratio=1.0,
    )

    # Eligible on first iteration
    assert refiner.can_refine(0, eval_insufficient) is True
    # Eligible on second iteration
    assert refiner.can_refine(1, eval_insufficient) is True
    # Ineligible when max reached
    assert refiner.can_refine(2, eval_insufficient) is False
    # Ineligible when already sufficient
    assert refiner.can_refine(0, eval_sufficient) is False


def test_refiner_targets_uncovered_subqueries():
    refiner = QueryRefiner()

    sq_cov1 = SubqueryCoverage(
        subquery_id="sq-1",
        subquery_text="What was revenue?",
        matched_chunk_ids=["c-1"],
        highest_similarity=0.88,
        is_covered=True,
    )
    sq_cov2 = SubqueryCoverage(
        subquery_id="sq-2",
        subquery_text="What were capital expenditures?",
        matched_chunk_ids=[],
        highest_similarity=0.0,
        is_covered=False,
    )

    eval_rep = EvidenceEvaluation(
        overall_score=0.44,
        is_sufficient=False,
        reason=SufficiencyReason.INSUFFICIENT_SUBQUERY_COVERAGE,
        explanation="CapEx missing",
        subquery_coverages=[sq_cov1, sq_cov2],
        coverage_ratio=0.50,
    )

    res = refiner.refine(
        current_iteration=0,
        original_query="What was revenue and CapEx?",
        evaluation=eval_rep,
        current_top_k=5,
        current_similarity_threshold=0.50,
    )

    assert res.iteration == 1
    assert res.strategy == RefinementStrategy.TARGET_UNCOVERED_SUBQUERIES
    assert "capital expenditures" in res.refined_query_text
    assert res.targeted_subquery_ids == ["sq-2"]
    assert res.adjusted_top_k == 7
    assert res.adjusted_similarity_threshold == 0.40


def test_refiner_relaxes_threshold_on_low_similarity():
    refiner = QueryRefiner()

    eval_rep = EvidenceEvaluation(
        overall_score=0.38,
        is_sufficient=False,
        reason=SufficiencyReason.LOW_SIMILARITY_SCORE,
        explanation="Low similarity",
        coverage_ratio=0.0,
    )

    res = refiner.refine(
        current_iteration=0,
        original_query="What is the internal encryption mechanism?",
        evaluation=eval_rep,
        current_top_k=5,
        current_similarity_threshold=0.60,
    )

    assert res.iteration == 1
    assert res.strategy == RefinementStrategy.RELAX_SIMILARITY_THRESHOLD
    assert res.adjusted_similarity_threshold == 0.45
    assert res.adjusted_top_k == 9


def test_refiner_broadens_scope_on_empty_retrieval():
    refiner = QueryRefiner()

    eval_rep = EvidenceEvaluation(
        overall_score=0.0,
        is_sufficient=False,
        reason=SufficiencyReason.EMPTY_RETRIEVAL,
        explanation="No chunks",
        coverage_ratio=0.0,
    )

    res = refiner.refine(
        current_iteration=0,
        original_query="What is XYZ-9000-alpha specification?!",
        evaluation=eval_rep,
        current_top_k=5,
    )

    assert res.iteration == 1
    assert res.strategy == RefinementStrategy.BROADEN_RETRIEVAL_SCOPE
    assert "XYZ-9000-alpha" in res.refined_query_text
    assert res.adjusted_top_k == 10


# ── 3. Controller Integration Tests with Adaptive Refinement ────────────────


@pytest.mark.asyncio
async def test_agent_controller_successful_adaptive_re_retrieval():
    col_id = str(uuid4())
    doc_id = str(uuid4())

    # Round 1 returns empty or low-scoring chunk
    empty_resp = MagicMock()
    empty_resp.chunks = []

    # Round 2 (after refinement) returns high-scoring relevant chunk
    chunk_round2 = make_chunk(
        "chunk-2",
        "The architecture consists of a plain Python state machine and SQLite storage.",
        similarity_score=0.92,
        doc_id=doc_id,
        col_id=col_id,
    )
    success_resp = MagicMock()
    success_resp.chunks = [chunk_round2]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    # 1st call: empty, 2nd call (re-retrieval): success
    mock_retrieval_svc.retrieve = AsyncMock(side_effect=[empty_resp, success_resp])

    rag_svc = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="The architecture uses a plain Python state machine [S1].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet",
            latency_ms=280.0,
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

    req = AgentRunRequest(query="Describe the system architecture.")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is True
    assert resp.iteration_count == 1
    assert len(resp.refinement_history) == 1
    assert resp.refinement_history[0].iteration == 1
    assert "plain Python state machine" in resp.answer

    # Verify event stream records QUERY_REFINED
    event_types = [e.event_type for e in resp.events]
    assert AgentEventType.QUERY_REFINED in event_types
    assert AgentEventType.EVIDENCE_INSUFFICIENT in event_types
    assert AgentEventType.EVIDENCE_EVALUATED in event_types


@pytest.mark.asyncio
async def test_agent_controller_refinement_loop_bounded_refusal():
    col_id = str(uuid4())

    # Persistent empty retrieval across all attempts
    empty_resp = MagicMock()
    empty_resp.chunks = []

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=empty_resp)

    rag_svc = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)

    controller = AgentController(
        rag_service=rag_svc,
        answer_generator=mock_answer_gen,
        query_refiner=QueryRefiner(max_refinements=2),
    )

    req = AgentRunRequest(query="Completely nonexistent information topic")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is False
    assert resp.iteration_count == 2
    assert len(resp.refinement_history) == 2
    assert resp.answer == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
    mock_answer_gen.generate_answer.assert_not_called()
