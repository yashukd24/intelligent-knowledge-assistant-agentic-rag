"""
tests/test_agent_decision_making.py
───────────────────────────────────────────────────────────────────────────────
Phase 6 Stage 2: Agent Decision-Making & Adaptive Retrieval Verification Suite.

Comprehensive tests verifying:
  1. Simple single-intent question resolving directly in 1 round.
  2. Complex multi-part question with subquery decomposition & unified synthesis.
  3. Missing evidence leading to safe refusal without LLM inference.
  4. Conflicting documents / version comparison with structured conflict guidance.
  5. Irrelevant retrieval results (borderline similarity + zero keyword overlap)
     preventing false-positive sufficiency decisions.
  6. Insufficient evidence in round 1 followed by adaptive refinement & success in round 2.
  7. Maximum retry exhaustion bounded strictly to 3 retrieval rounds (1 initial + 2 refinements).
  8. Invalid citations detection & hallucination flagging via AnswerVerifier.
  9. Safe abstention ensuring zero LLM tokens and standard refusal on unanswerable queries.
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
    EvidenceEvaluation,
    SufficiencyReason,
)
from backend.schemas.query_plan import QueryIntent, SubQuery
from backend.schemas.query_refinement import RefinementStrategy
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
from backend.services.query_refiner import QueryRefiner
from backend.services.rag_service import RAGService
from backend.services.retrieval_service import RetrievalService


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


# ── 1. Simple Question Test ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_question_direct_resolution():
    """Verify that a simple factual query resolves cleanly in 1 retrieval round."""
    col_id = str(uuid4())
    doc_id = str(uuid4())

    chunk = make_chunk(
        "chunk-simple-1",
        "The CSE HOD Cabin is located in Room 302 on the 3rd Floor of Block A.",
        similarity_score=0.91,
        doc_id=doc_id,
        col_id=col_id,
    )
    retrieval_resp = MagicMock()
    retrieval_resp.chunks = [chunk]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=retrieval_resp)

    rag_svc = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="The CSE HOD cabin is located in Room 302, Block A [S1].",
            status=AnswerStatus.ANSWERED,
            provider="mock",
            model="mock-model",
            latency_ms=120.0,
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

    req = AgentRunRequest(query="Where is the CSE HOD cabin?")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is True
    assert resp.iteration_count == 0  # 0 refinements = exactly 1 retrieval round
    assert len(resp.refinement_history) == 0
    assert "Room 302" in resp.answer
    assert "[S1]" in resp.valid_citations
    assert mock_retrieval_svc.retrieve.call_count == 1
    mock_answer_gen.generate_answer.assert_called_once()


# ── 2. Complex Multi-Part Question Test ───────────────────────────────────────


@pytest.mark.asyncio
async def test_complex_multi_part_question_decomposition():
    """Verify multi-subquery decomposition and synthesized cross-chunk answer."""
    col_id = str(uuid4())
    doc_id = str(uuid4())

    chunk1 = make_chunk(
        "chunk-complex-1",
        "Placement eligibility requires a minimum aggregate of 65.0% and zero active backlogs.",
        similarity_score=0.88,
        doc_id=doc_id,
        col_id=col_id,
    )
    chunk2 = make_chunk(
        "chunk-complex-2",
        "Attendance requirement for semester exams is strictly 75% in each enrolled course.",
        similarity_score=0.86,
        doc_id=doc_id,
        col_id=col_id,
    )

    resp_sq1 = MagicMock()
    resp_sq1.chunks = [chunk1]

    resp_sq2 = MagicMock()
    resp_sq2.chunks = [chunk2]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(side_effect=[resp_sq1, resp_sq2])

    rag_svc = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="Placement requires 65% aggregate [S1], and attendance requirement is 75% [S2].",
            status=AnswerStatus.ANSWERED,
            provider="mock",
            model="mock-model",
            latency_ms=250.0,
            citations=CitationValidationResult(
                cited_sources=["[S1]", "[S2]"],
                valid_citations=["[S1]", "[S2]"],
            ),
        )
    )

    controller = AgentController(
        rag_service=rag_svc,
        answer_generator=mock_answer_gen,
    )

    compound_query = "What is the placement eligibility criteria, and what is the minimum attendance requirement?"
    req = AgentRunRequest(query=compound_query)
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is True
    assert len(resp.decomposed_queries) == 2
    assert len(resp.sources) == 2
    assert "[S1]" in resp.valid_citations
    assert "[S2]" in resp.valid_citations
    assert mock_retrieval_svc.retrieve.call_count == 2


# ── 3. Missing Evidence Test ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_evidence_safe_refusal():
    """Verify that zero retrieved evidence leads directly to safe refusal without LLM call."""
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
    )

    req = AgentRunRequest(query="What is the quantum telemetry satellite bandwidth?")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is False
    assert resp.answer == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
    # LLM must NEVER be called when evidence is missing
    mock_answer_gen.generate_answer.assert_not_called()


# ── 4. Conflicting Documents / Version Comparison Test ───────────────────────


@pytest.mark.asyncio
async def test_conflicting_documents_comparison_guidance():
    """Verify that comparative / multi-version queries inject structured conflict resolution instructions."""
    col_id = str(uuid4())

    chunk_v1 = make_chunk(
        "chunk-v1",
        "Placement Policy v1.0: Minimum aggregate eligibility is 65% with zero backlogs.",
        similarity_score=0.87,
        doc_id=str(uuid4()),
        col_id=col_id,
    )
    chunk_v2 = make_chunk(
        "chunk-v2",
        "Placement Policy Draft v2.0: Proposed aggregate eligibility increased to 70% with zero backlogs.",
        similarity_score=0.85,
        doc_id=str(uuid4()),
        col_id=col_id,
    )

    retrieval_resp = MagicMock()
    retrieval_resp.chunks = [chunk_v1, chunk_v2]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=retrieval_resp)

    prompt_captured = {}

    async def mock_generate_answer(prompt, **kwargs):
        prompt_captured["prompt"] = prompt
        return AnswerGenerationResult(
            answer="Policy v1.0 requires 65% aggregate [S1], whereas Draft v2.0 proposes 70% [S2].",
            status=AnswerStatus.ANSWERED,
            provider="mock",
            model="mock-model",
            latency_ms=200.0,
            citations=CitationValidationResult(
                cited_sources=["[S1]", "[S2]"],
                valid_citations=["[S1]", "[S2]"],
            ),
        )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(side_effect=mock_generate_answer)

    controller = AgentController(
        rag_service=RAGService(
            retrieval_service=mock_retrieval_svc,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
        ),
        answer_generator=mock_answer_gen,
    )

    query = "Compare placement policy v1.0 with draft v2.0 requirements."
    req = AgentRunRequest(query=query)
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.comparison_objective is not None
    # Verify the prompt contained the comparison and conflict resolution instructions
    assert "=== COMPARISON & CONFLICT RESOLUTION ===" in prompt_captured["prompt"]
    assert "Placement Policy v1.0" in prompt_captured["prompt"]
    assert "Placement Policy Draft v2.0" in prompt_captured["prompt"]


# ── 5. Irrelevant Retrieval Results / False-Positive Rejection ───────────────


def test_irrelevant_retrieval_prevents_false_positive_sufficiency():
    """Verify that chunks with borderline similarity but 0 keyword overlap are rejected as qualifying."""
    evaluator = EvidenceEvaluator(default_similarity_threshold=0.50)

    # Multi-term query with specific keywords: "hostel", "fee", "refund", "policy"
    query = "What is the hostel fee refund policy?"

    # Irrelevant chunk with borderline similarity (0.52) but completely unrelated vocabulary
    chunk = make_chunk(
        "chunk-noise-1",
        "The computer science laboratory has 60 workstations equipped with Linux and Python.",
        similarity_score=0.52,
    )

    eval_result = evaluator.evaluate(
        query=query,
        chunks=[chunk],
        assembled_context=chunk.text,
        similarity_threshold=0.50,
    )

    # Must NOT declare sufficient evidence for completely unrelated content
    assert eval_result.is_sufficient is False
    assert eval_result.qualifying_chunk_count == 0
    assert eval_result.reason == SufficiencyReason.NO_RELEVANT_CONTENT


# ── 6. Insufficient Evidence Followed by Successful Refinement ───────────────


@pytest.mark.asyncio
async def test_adaptive_refinement_recovers_missing_evidence():
    """Verify that round 1 insufficient evidence triggers refinement, recovers evidence in round 2, and succeeds."""
    col_id = str(uuid4())

    # Round 1: Low similarity / no qualifying content
    chunk_low = make_chunk(
        "chunk-low",
        "General campus amenities include sports grounds and student union spaces.",
        similarity_score=0.38,
        col_id=col_id,
    )
    resp_r1 = MagicMock()
    resp_r1.chunks = [chunk_low]

    # Round 2: Targeted refined retrieval returns exact document
    chunk_refined = make_chunk(
        "chunk-high",
        "Internship guidelines mandate an 8-week summer internship approved via faculty NOC.",
        similarity_score=0.92,
        col_id=col_id,
    )
    resp_r2 = MagicMock()
    resp_r2.chunks = [chunk_refined]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(side_effect=[resp_r1, resp_r2])

    prompt_delivered = {}

    async def mock_generate(prompt, **kwargs):
        prompt_delivered["prompt"] = prompt
        return AnswerGenerationResult(
            answer="Summer internships must be 8 weeks long and require a faculty NOC [S1].",
            status=AnswerStatus.ANSWERED,
            provider="mock",
            model="mock-model",
            latency_ms=180.0,
            citations=CitationValidationResult(
                cited_sources=["[S1]"],
                valid_citations=["[S1]"],
            ),
        )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(side_effect=mock_generate)

    controller = AgentController(
        rag_service=RAGService(
            retrieval_service=mock_retrieval_svc,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
        ),
        answer_generator=mock_answer_gen,
        query_refiner=QueryRefiner(max_refinements=2),
    )

    req = AgentRunRequest(query="What are the summer internship duration guidelines?")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is True
    assert resp.iteration_count == 1  # 1 refinement occurred
    assert len(resp.refinement_history) == 1
    # Verify that the rebuilt prompt delivered to the LLM contains the newly retrieved chunk!
    assert "8-week summer internship" in prompt_delivered["prompt"]
    assert "NOC" in prompt_delivered["prompt"]


# ── 7. Maximum Retry Exhaustion Bounded to 3 Retrieval Rounds ─────────────────


@pytest.mark.asyncio
async def test_maximum_retry_exhaustion_strictly_bounded():
    """Verify that persistent missing evidence strictly stops after 2 refinements (3 rounds total)."""
    empty_resp = MagicMock()
    empty_resp.chunks = []

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=empty_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)

    controller = AgentController(
        rag_service=RAGService(
            retrieval_service=mock_retrieval_svc,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
        ),
        answer_generator=mock_answer_gen,
        query_refiner=QueryRefiner(max_refinements=2),
    )

    req = AgentRunRequest(query="Completely nonexistent policy details")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is False
    # Max refinements = 2; total retrieval rounds = 1 initial + 2 refinements = 3
    assert resp.iteration_count == 2
    assert len(resp.refinement_history) == 2
    assert resp.answer == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
    # Exactly 3 retrieval calls were made (1 initial + 2 adaptive re-retrievals)
    assert mock_retrieval_svc.retrieve.call_count == 3
    # LLM inference never executed
    mock_answer_gen.generate_answer.assert_not_called()


# ── 8. Invalid Citations Detection Test ───────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_citations_detection_and_flagging():
    """Verify that hallucinated citations (e.g. [S99]) are identified and flagged."""
    col_id = str(uuid4())
    chunk = make_chunk(
        "chunk-cites",
        "Tuition fee must be paid before the 15th of August each academic year.",
        similarity_score=0.90,
        col_id=col_id,
    )
    resp_retrieval = MagicMock()
    resp_retrieval.chunks = [chunk]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=resp_retrieval)

    # LLM hallucinates citation [S99] which was never in the sources
    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="Tuition fee is due on August 15th [S1], and late fee applies [S99].",
            status=AnswerStatus.ANSWERED,
            provider="mock",
            model="mock-model",
            latency_ms=190.0,
            citations=CitationValidationResult(
                cited_sources=["[S1]", "[S99]"],
                valid_citations=["[S1]"],
                invalid_citations=["[S99]"],
            ),
        )
    )

    controller = AgentController(
        rag_service=RAGService(
            retrieval_service=mock_retrieval_svc,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
        ),
        answer_generator=mock_answer_gen,
    )

    req = AgentRunRequest(query="When is the tuition fee due?")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert "[S1]" in resp.valid_citations
    assert "[S99]" in resp.invalid_citations
    assert resp.verification is not None
    assert "[S99]" in resp.verification.invalid_citations
    assert resp.verification.hallucination_warning is True


# ── 9. Safe Abstention Test ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_safe_abstention_zero_llm_inference():
    """Verify that an unanswerable query produces safe abstention with zero token usage."""
    empty_resp = MagicMock()
    empty_resp.chunks = []

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=empty_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)

    controller = AgentController(
        rag_service=RAGService(
            retrieval_service=mock_retrieval_svc,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
        ),
        answer_generator=mock_answer_gen,
    )

    req = AgentRunRequest(query="What is the confidential salary of the university dean?")
    resp = await controller.run(request=req, collection_id=uuid4())

    assert resp.status == AgentStatus.COMPLETED
    assert resp.is_sufficient is False
    assert resp.answer == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
    # Verified: zero LLM inference, clean trace
    mock_answer_gen.generate_answer.assert_not_called()
    assert any(e.event_type == AgentEventType.EVIDENCE_INSUFFICIENT for e in resp.events)
    assert any(e.event_type == AgentEventType.AGENT_DONE for e in resp.events)
