"""
tests/evaluation/test_retrieval_evaluator.py
─────────────────────────────────────────────────────────────
Comprehensive tests for RetrievalEvaluator:
  - Evaluation dataset structure and query types.
  - Pydantic schema validation.
  - Deterministic evaluation using mock RetrievalService.
  - Error isolation and failure handling.
  - Report JSON serialization and disk saving.
  - End-to-end integration test with isolated SQLite and ChromaDB.
─────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.config import Settings
from backend.database import setup_sqlite_foreign_keys
from backend.evaluation.evaluation_dataset import (
    EVAL_COLLECTION_ID,
    SYNTHETIC_TEST_CASES,
    load_evaluation_dataset,
    seed_evaluation_dataset,
)
from backend.evaluation.evaluation_models import (
    EvaluationReport,
    EvaluationTestCase,
    QueryEvaluationResult,
)
from backend.evaluation.retrieval_evaluator import RetrievalEvaluator
from backend.models.base import Base
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse, RetrievedChunk
from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval_service import (
    RetrievalExecutionError,
    RetrievalService,
)
from backend.services.vector_store import VectorStoreService


# ── Dataset & Schema Unit Tests ──────────────────────────────────────────────


def test_evaluation_dataset_loading_and_query_types():
    """Validates that synthetic dataset contains >= 10 cases covering all required query types."""
    dataset = load_evaluation_dataset()
    assert len(dataset) >= 10

    required_types = {
        "factual",
        "conceptual",
        "definition",
        "multi-part",
        "cross-chunk",
        "paraphrase",
        "limited-info",
        "unanswerable",
        "similar-terms",
        "doc-specific",
    }
    present_types = {tc.query_type for tc in dataset}
    assert required_types.issubset(present_types), f"Missing query types: {required_types - present_types}"

    # Verify each case has valid fields
    for tc in dataset:
        assert tc.case_id
        assert tc.query
        assert tc.collection_id == EVAL_COLLECTION_ID
        assert isinstance(tc.expected_chunk_ids, list)


def test_evaluation_test_case_validation():
    """Validates Pydantic enforcement on EvaluationTestCase."""
    # Valid
    tc = EvaluationTestCase(
        case_id="tc_valid",
        query="What is the policy?",
        collection_id=EVAL_COLLECTION_ID,
        expected_chunk_ids=["chunk_1"],
    )
    assert tc.case_id == "tc_valid"

    # Empty case_id raises ValueError
    with pytest.raises(ValueError):
        EvaluationTestCase(
            case_id="   ",
            query="Valid query",
            collection_id=EVAL_COLLECTION_ID,
            expected_chunk_ids=["c1"],
        )

    # Empty query raises ValueError
    with pytest.raises(ValueError):
        EvaluationTestCase(
            case_id="tc_1",
            query="",
            collection_id=EVAL_COLLECTION_ID,
            expected_chunk_ids=["c1"],
        )


# ── Deterministic Mock Evaluation Tests ──────────────────────────────────────


def make_dummy_chunk(chunk_id: str, doc_id: str = "doc_1", score: float = 0.90) -> RetrievedChunk:
    """Helper to build a candidate chunk for mock evaluation."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        collection_id=EVAL_COLLECTION_ID,
        text="Sample test text",
        similarity_score=score,
        distance=1.0 - score,
        chunk_index=0,
        version=1,
    )


@pytest.mark.asyncio
async def test_deterministic_mock_evaluation():
    """Validates metrics aggregation using a deterministic mock RetrievalService."""
    # 2 Test cases
    tc1 = EvaluationTestCase(
        case_id="tc_1",
        query="Question 1",
        collection_id=EVAL_COLLECTION_ID,
        expected_chunk_ids=["c1", "c2"],
        query_type="factual",
    )
    tc2 = EvaluationTestCase(
        case_id="tc_2",
        query="Question 2",
        collection_id=EVAL_COLLECTION_ID,
        expected_chunk_ids=["c3"],
        query_type="definition",
    )

    # Mock service
    mock_service = MagicMock(spec=RetrievalService)

    # Query 1 returns c1, x1 (1 relevant out of 2 expected, cutoff k=2)
    resp1 = RetrievalResponse(
        query="Question 1",
        collection_id=EVAL_COLLECTION_ID,
        results=[make_dummy_chunk("c1"), make_dummy_chunk("x1")],
        total_candidates=4,
        returned_results=2,
        rejected_candidates=1,
        accepted_candidates=2,
    )

    # Query 2 returns c3 (1 relevant out of 1 expected, cutoff k=2)
    resp2 = RetrievalResponse(
        query="Question 2",
        collection_id=EVAL_COLLECTION_ID,
        results=[make_dummy_chunk("c3")],
        total_candidates=2,
        returned_results=1,
        rejected_candidates=0,
        accepted_candidates=1,
    )

    mock_service.retrieve = AsyncMock(side_effect=[resp1, resp2])
    evaluator = RetrievalEvaluator(retrieval_service=mock_service, dataset=[tc1, tc2])

    report = await evaluator.run_evaluation(top_k=2)

    assert isinstance(report, EvaluationReport)
    assert report.dataset_size == 2
    agg = report.aggregate_metrics

    # For tc1: retrieved=["c1", "x1"], expected=["c1", "c2"]
    # P@2 = 1/2 = 0.5, R@2 = 1/2 = 0.5, RR = 1.0 (c1 at rank 1), Hit = True
    # For tc2: retrieved=["c3"], expected=["c3"]
    # P@2 = 1/2 = 0.5, R@2 = 1/1 = 1.0, RR = 1.0 (c3 at rank 1), Hit = True
    # Mean P@2 = (0.5 + 0.5)/2 = 0.5
    # Mean R@2 = (0.5 + 1.0)/2 = 0.75
    # MRR = 1.0
    # Hit Rate = 1.0
    assert agg.mean_precision_at_k == 0.5
    assert agg.mean_recall_at_k == 0.75
    assert agg.mrr == 1.0
    assert agg.hit_rate_at_k == 1.0
    assert agg.empty_result_count == 0
    assert agg.total_candidates_sum == 6
    assert agg.accepted_candidates_sum == 3
    assert agg.rejected_candidates_sum == 1
    assert "factual" in agg.by_query_type
    assert "definition" in agg.by_query_type


@pytest.mark.asyncio
async def test_evaluation_failure_handling():
    """Evaluator captures exceptions gracefully without terminating the entire run."""
    tc1 = EvaluationTestCase(
        case_id="tc_fail",
        query="Broken query",
        collection_id=EVAL_COLLECTION_ID,
        expected_chunk_ids=["c1"],
    )

    mock_service = MagicMock(spec=RetrievalService)
    mock_service.retrieve = AsyncMock(side_effect=RetrievalExecutionError("ChromaDB read fault"))

    evaluator = RetrievalEvaluator(retrieval_service=mock_service, dataset=[tc1])
    report = await evaluator.run_evaluation(top_k=5)

    assert report.dataset_size == 1
    assert report.aggregate_metrics.failed_queries == 1
    assert report.aggregate_metrics.successful_queries == 0
    assert len(report.failed_cases) == 1
    assert "ChromaDB read fault" in report.failed_cases[0]["error"]
    assert report.query_results[0].error is not None


def test_evaluation_report_json_serialization(tmp_path: Path):
    """Validates report serialization to JSON string and file writing."""
    tc = EvaluationTestCase(
        case_id="tc_1", query="Q", collection_id=EVAL_COLLECTION_ID, expected_chunk_ids=["c1"]
    )
    res = QueryEvaluationResult(
        case_id="tc_1",
        query="Q",
        query_type="factual",
        retrieved_chunk_ids=["c1"],
        expected_chunk_ids=["c1"],
        relevant_retrieved_chunk_ids=["c1"],
        precision_at_k=1.0,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        hit=True,
    )
    from backend.evaluation.evaluation_models import AggregateMetrics

    agg = AggregateMetrics(
        mean_precision_at_k=1.0,
        mean_recall_at_k=1.0,
        mrr=1.0,
        hit_rate_at_k=1.0,
        empty_result_count=0,
        empty_result_rate=0.0,
        avg_latency_ms=10.0,
        median_latency_ms=10.0,
        p95_latency_ms=10.0,
        total_candidates_sum=5,
        rejected_candidates_sum=0,
        accepted_candidates_sum=5,
        total_queries=1,
        successful_queries=1,
        failed_queries=0,
    )

    report = EvaluationReport(
        dataset_name="test-ds",
        dataset_size=1,
        top_k=5,
        query_results=[res],
        aggregate_metrics=agg,
    )

    # String serialization
    json_str = report.to_json()
    assert "mean_precision_at_k" in json_str
    assert '"top_k": 5' in json_str

    # Disk file writing
    out_file = tmp_path / "reports" / "report.json"
    saved = report.save_json(out_file)
    assert saved.exists()
    assert "test-ds" in saved.read_text(encoding="utf-8")


# ── Integration Test with Isolated Resources ────────────────────────────────


@pytest.mark.asyncio
async def test_end_to_end_isolated_evaluation(tmp_path: Path):
    """
    End-to-end integration test:
      1. Creates an isolated in-memory SQLite database.
      2. Creates a temporary ChromaDB directory.
      3. Seeds the synthetic university policy dataset (4 documents, 10 chunks).
      4. Executes RetrievalEvaluator across all 10 evaluation test cases.
      5. Validates non-zero precision, recall, MRR, and clean candidate accounting.
    """
    chroma_dir = tmp_path / "eval_chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db_url = "sqlite+aiosqlite:///:memory:"

    eval_settings = Settings(
        database_url=db_url,
        chroma_persist_dir=str(chroma_dir),
        log_level="WARNING",
    )

    engine = create_async_engine(db_url, connect_args={"check_same_thread": False}, future=True)
    setup_sqlite_foreign_keys(engine)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    vector_store = VectorStoreService(persist_dir=str(chroma_dir), settings=eval_settings)
    embedding_service = EmbeddingService(settings=eval_settings)

    # Seed isolated dataset
    async with factory() as session:
        col_id = await seed_evaluation_dataset(
            db=session,
            vector_store=vector_store,
            embedding_service=embedding_service,
        )

    dataset = load_evaluation_dataset(collection_id=col_id)
    assert len(dataset) == 10

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        session_factory=factory,
        settings=eval_settings,
    )

    evaluator = RetrievalEvaluator(
        retrieval_service=retrieval_service,
        dataset=dataset,
        session_factory=factory,
        dataset_name="synthetic-university-policies-v1",
    )

    # Run evaluation with top_k=5
    report = await evaluator.run_evaluation(top_k=5)

    assert isinstance(report, EvaluationReport)
    assert report.dataset_size == 10
    agg = report.aggregate_metrics

    # Verify execution metrics
    assert agg.total_queries == 10
    assert agg.successful_queries == 10
    assert agg.failed_queries == 0
    assert agg.hit_rate_at_k > 0.70  # Semantic retrieval achieves high hit rate
    assert agg.mrr > 0.60           # Relevant chunks ranked near top
    assert agg.mean_recall_at_k > 0.60
    assert agg.avg_latency_ms > 0.0

    await engine.dispose()
