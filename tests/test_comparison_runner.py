"""
tests/test_comparison_runner.py
─────────────────────────────────────────────────────────────
Phase 20: Unit and API tests for ComparativeEvaluator and
Evaluation Router (/api/evaluation/compare, /api/evaluation/metrics).
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient

from backend.evaluation.comparison_runner import (
    ComparativeEvaluationReport,
    ComparativeEvaluator,
    PipelineMetrics,
)
from backend.evaluation.evaluation_models import EvaluationTestCase
from backend.main import app
from backend.routers.agent import get_agent_controller
from backend.routers.rag import get_rag_service
from backend.schemas.agent_state import AgentRunResponse, AgentStatus
from backend.schemas.rag import RAGResponse, SourceReference
from backend.schemas.synthesis import SynthesisVerification


@pytest.fixture
def sample_test_cases():
    return [
        EvaluationTestCase(
            case_id="case_1",
            query="What is the refund policy?",
            collection_id=str(uuid4()),
            query_type="factual",
            difficulty="easy",
            expected_chunk_ids=["chunk_refund_1", "chunk_refund_2"],
        ),
        EvaluationTestCase(
            case_id="case_2",
            query="Compare grading criteria across courses.",
            collection_id=str(uuid4()),
            query_type="comparative",
            difficulty="medium",
            expected_chunk_ids=["chunk_grading_a"],
        ),
    ]


@pytest.mark.asyncio
async def test_comparative_evaluator_metrics_computation(sample_test_cases):
    # Setup mock RAGService
    mock_rag_service = MagicMock()
    # Case 1 returns chunk_refund_1 (Hit, MRR=1.0)
    # Case 2 returns chunk_unrelated (Miss, MRR=0.0)
    mock_rag_resp_1 = RAGResponse(
        query=sample_test_cases[0].query,
        collection_id=str(sample_test_cases[0].collection_id),
        retrieved_chunk_count=1,
        assembled_context="Policy text",
        context_character_count=100,
        prompt_text="Prompt text",
        sources=[
            SourceReference(
                source_id="[S1]",
                chunk_id="chunk_refund_1",
                document_id=str(uuid4()),
                chunk_index=0,
                page_number=1,
                similarity_score=0.88,
                char_count=50,
            )
        ],
    )
    mock_rag_resp_2 = RAGResponse(
        query=sample_test_cases[1].query,
        collection_id=str(sample_test_cases[1].collection_id),
        retrieved_chunk_count=1,
        assembled_context="Other text",
        context_character_count=80,
        prompt_text="Prompt text",
        sources=[
            SourceReference(
                source_id="[S1]",
                chunk_id="chunk_other",
                document_id=str(uuid4()),
                chunk_index=0,
                page_number=1,
                similarity_score=0.60,
                char_count=40,
            )
        ],
    )
    mock_rag_service.build_rag_context = AsyncMock(side_effect=[mock_rag_resp_1, mock_rag_resp_2])

    # Setup mock AgentController
    mock_agent_controller = MagicMock()
    mock_agent_resp_1 = AgentRunResponse(
        session_id="sess_1",
        trace_id="tr_1",
        collection_id=str(sample_test_cases[0].collection_id),
        query=sample_test_cases[0].query,
        status=AgentStatus.COMPLETED,
        answer="The refund policy is 100% within 14 days.",
        sources=[
            SourceReference(
                source_id="[S1]",
                chunk_id="chunk_refund_1",
                document_id=str(uuid4()),
                chunk_index=0,
                page_number=1,
                similarity_score=0.92,
                char_count=50,
            ),
            SourceReference(
                source_id="[S2]",
                chunk_id="chunk_refund_2",
                document_id=str(uuid4()),
                chunk_index=1,
                page_number=2,
                similarity_score=0.90,
                char_count=50,
            ),
        ],
        citations=[],
        evidence_score=0.92,
        is_sufficient=True,
        refinement_history=[],
        verification=SynthesisVerification(
            is_verified=True,
            grounding_score=0.95,
            total_sentences=1,
            grounded_sentences=1,
            valid_citations=["[S1]"],
            invalid_citations=[],
            uncited_sources=[],
            sentence_evaluations=[],
            hallucination_warning=False,
            grounding_explanation="All claims fully grounded in citations.",
        ),
        total_latency_ms=45.0,
    )
    mock_agent_resp_2 = AgentRunResponse(
        session_id="sess_2",
        trace_id="tr_2",
        collection_id=str(sample_test_cases[1].collection_id),
        query=sample_test_cases[1].query,
        status=AgentStatus.COMPLETED,
        answer="Grading criteria comparison.",
        sources=[
            SourceReference(
                source_id="[S1]",
                chunk_id="chunk_grading_a",
                document_id=str(uuid4()),
                chunk_index=0,
                page_number=1,
                similarity_score=0.85,
                char_count=40,
            )
        ],
        citations=[],
        evidence_score=0.85,
        is_sufficient=True,
        refinement_history=[],
        verification=SynthesisVerification(
            is_verified=True,
            grounding_score=0.90,
            total_sentences=1,
            grounded_sentences=1,
            valid_citations=["[S1]"],
            invalid_citations=[],
            uncited_sources=[],
            sentence_evaluations=[],
            hallucination_warning=False,
            grounding_explanation="All claims fully grounded in citations.",
        ),
        total_latency_ms=80.0,
    )
    mock_agent_controller.run = AsyncMock(side_effect=[mock_agent_resp_1, mock_agent_resp_2])

    evaluator = ComparativeEvaluator(
        rag_service=mock_rag_service,
        agent_controller=mock_agent_controller,
    )

    report = await evaluator.evaluate_dataset(sample_test_cases, dataset_name="test-suite")

    assert isinstance(report, ComparativeEvaluationReport)
    assert report.dataset_name == "test-suite"
    assert report.dataset_size == 2

    # Conventional metrics: 1 hit out of 2 = 0.5 hit rate
    assert report.conventional.hit_rate_at_5 == 0.5
    assert report.conventional.mrr == 0.5
    assert report.conventional.total_queries == 2
    assert report.conventional.successful_queries == 1

    # Agentic metrics: 2 hits out of 2 = 1.0 hit rate
    assert report.agentic.hit_rate_at_5 == 1.0
    assert report.agentic.mrr == 1.0
    assert report.agentic.total_queries == 2
    assert report.agentic.successful_queries == 2
    assert report.agentic.grounding_score > 0.90

    # Improvements
    assert report.relative_improvements["hit_rate_delta"] == 0.5
    assert report.relative_improvements["mrr_delta"] == 0.5
    assert "Phase 20 comparative evaluation" in report.research_summary


@pytest.mark.asyncio
async def test_comparative_evaluator_error_isolation(sample_test_cases):
    """Verifies that an exception during execution of one query does not crash the evaluation."""
    mock_rag_service = MagicMock()
    mock_rag_service.build_rag_context = AsyncMock(side_effect=RuntimeError("Vector DB unavailable"))

    mock_agent_controller = MagicMock()
    mock_agent_controller.run = AsyncMock(side_effect=ValueError("LLM parse error"))

    evaluator = ComparativeEvaluator(
        rag_service=mock_rag_service,
        agent_controller=mock_agent_controller,
    )

    report = await evaluator.evaluate_dataset(sample_test_cases, dataset_name="failing-suite")
    assert report.conventional.successful_queries == 0
    assert report.conventional.hit_rate_at_5 == 0.0
    assert report.agentic.successful_queries == 0
    assert report.agentic.hit_rate_at_5 == 0.0


@pytest.mark.asyncio
async def test_evaluation_router_metrics_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/evaluation/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "dataset_name" in data
        assert "aggregate_metrics" in data
        metrics = data["aggregate_metrics"]
        assert "hit_rate_at_k" in metrics
        assert "mrr" in metrics
        assert "avg_latency_ms" in metrics
