"""
tests/test_rag_answer_service.py
─────────────────────────────────────────────────────────────
Comprehensive unit and orchestration tests for RAGAnswerService.
Verifies integration between RAGService, AnswerGenerator,
and SQLite collection validation.
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.collection import Collection
from backend.schemas.rag import RAGResponse, SourceReference
from backend.schemas.rag_answer import (
    RAGAnswerRequest,
    RAGAnswerResponse,
)
from backend.services.answer_generator import (
    AnswerGenerationResult,
    AnswerGenerator,
    AnswerStatus,
)
from backend.services.citation_validator import CitationValidationResult
from backend.services.llm_provider import TokenUsage
from backend.services.rag_answer_service import RAGAnswerService
from backend.services.rag_service import RAGExecutionError, RAGService
from backend.services.vector_store import CollectionNotFoundError


async def _create_test_collection(db: AsyncSession) -> Collection:
    col_id = str(uuid4())
    col = Collection(
        id=col_id,
        name=f"Test Q&A Collection {col_id[:8]}",
        description="Collection for testing RAG answer service",
    )
    db.add(col)
    await db.commit()
    await db.refresh(col)
    return col


@pytest.mark.asyncio
async def test_rag_answer_service_successful_answer(db_session: AsyncSession):
    col = await _create_test_collection(db_session)
    col_uuid = UUID(col.id)

    src = SourceReference(
        source_id="[S1]",
        document_id=str(uuid4()),
        chunk_id=str(uuid4()),
        filename="paper.pdf",
        page_number=3,
        chunk_index=2,
        similarity_score=0.91,
        char_count=100,
    )
    mock_rag_resp = RAGResponse(
        query="What is transformer?",
        collection_id=col.id,
        retrieved_chunk_count=3,
        assembled_context="[Source 1]: Transformers are powerful.",
        context_character_count=40,
        sources=[src],
        prompt_text="Context: [Source 1] Transformers. Question: What is transformer?",
        excluded_chunks=[],
        retrieval_metadata={"candidate_count": 3},
    )

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(return_value=mock_rag_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="Transformers are neural network models [S1].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=TokenUsage(prompt_tokens=150, completion_tokens=30, total_tokens=180),
            latency_ms=450.0,
            citations=CitationValidationResult(
                cited_sources=["[S1]"],
                valid_citations=["[S1]"],
                invalid_citations=[],
                uncited_sources=[],
                valid_source_references=[src],
                has_citations=True,
                has_invalid_citations=False,
            ),
            error_message=None,
        )
    )

    svc = RAGAnswerService(rag_service=mock_rag_svc, answer_generator=mock_answer_gen)
    request = RAGAnswerRequest(
        query="What is transformer?",
        top_k=5,
        max_context_chars=12000,
    )

    resp: RAGAnswerResponse = await svc.answer_question(
        collection_id=col_uuid,
        request=request,
        db=db_session,
    )

    assert resp.status == AnswerStatus.ANSWERED
    assert "Transformers are neural network models [S1]" in resp.answer
    assert resp.collection_id == col.id
    assert len(resp.sources) == 1
    assert resp.valid_citations == ["[S1]"]
    assert resp.invalid_citations == []
    assert resp.uncited_sources == []
    assert resp.model == "claude-3-5-sonnet-20241022"
    assert resp.usage.total_tokens == 180
    assert resp.error_message is None


@pytest.mark.asyncio
async def test_rag_answer_service_insufficient_evidence(db_session: AsyncSession):
    col = await _create_test_collection(db_session)
    col_uuid = UUID(col.id)

    empty_rag_resp = RAGResponse(
        query="What is quantum gravity?",
        collection_id=col.id,
        retrieved_chunk_count=0,
        assembled_context="",
        context_character_count=0,
        sources=[],
        prompt_text="",
        excluded_chunks=[],
        retrieval_metadata={"candidate_count": 0},
    )

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(return_value=empty_rag_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="The provided documents do not contain sufficient information to answer this question.",
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            provider="rule_based",
            model="none",
            usage=None,
            latency_ms=0.0,
            citations=CitationValidationResult(),
            error_message=None,
        )
    )

    svc = RAGAnswerService(rag_service=mock_rag_svc, answer_generator=mock_answer_gen)
    request = RAGAnswerRequest(query="What is quantum gravity?")

    resp: RAGAnswerResponse = await svc.answer_question(
        collection_id=col_uuid,
        request=request,
        db=db_session,
    )

    assert resp.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "sufficient information" in resp.answer
    assert resp.sources == []
    assert resp.valid_citations == []


@pytest.mark.asyncio
async def test_rag_answer_service_collection_not_found(db_session: AsyncSession):
    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(side_effect=CollectionNotFoundError("Collection not found"))
    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    svc = RAGAnswerService(rag_service=mock_rag_svc, answer_generator=mock_answer_gen)

    random_id = uuid4()
    request = RAGAnswerRequest(query="Any question")

    with pytest.raises(CollectionNotFoundError, match="Collection not found"):
        await svc.answer_question(
            collection_id=random_id,
            request=request,
            db=db_session,
        )


@pytest.mark.asyncio
async def test_rag_answer_service_rag_error_raises_exception(db_session: AsyncSession):
    col = await _create_test_collection(db_session)
    col_uuid = UUID(col.id)

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(
        side_effect=RAGExecutionError("Underlying vector search failed")
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    svc = RAGAnswerService(rag_service=mock_rag_svc, answer_generator=mock_answer_gen)

    with pytest.raises(RAGExecutionError, match="Underlying vector search failed"):
        await svc.answer_question(
            collection_id=col_uuid,
            request=RAGAnswerRequest(query="Test question"),
            db=db_session,
        )


@pytest.mark.asyncio
async def test_rag_answer_service_execution_trace_structure(db_session: AsyncSession):
    col = await _create_test_collection(db_session)
    col_uuid = UUID(col.id)

    src = SourceReference(
        source_id="[S1]",
        document_id=str(uuid4()),
        chunk_id=str(uuid4()),
        filename="system_spec.pdf",
        page_number=1,
        chunk_index=0,
        similarity_score=0.92,
        char_count=120,
    )
    mock_rag_resp = RAGResponse(
        query="Explain system architecture",
        collection_id=col.id,
        retrieved_chunk_count=1,
        assembled_context="[Source 1]: System architecture details.",
        context_character_count=42,
        sources=[src],
        prompt_text="Context: [Source 1] System architecture. Question: Explain system architecture",
        excluded_chunks=[],
        retrieval_metadata={"candidate_count": 1},
    )

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(return_value=mock_rag_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="The system employs a layered RAG architecture [S1].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=TokenUsage(prompt_tokens=120, completion_tokens=25, total_tokens=145),
            latency_ms=320.0,
            citations=CitationValidationResult(
                cited_sources=["[S1]"],
                valid_citations=["[S1]"],
                invalid_citations=[],
                uncited_sources=[],
                valid_source_references=[src],
                has_citations=True,
                has_invalid_citations=False,
            ),
            error_message=None,
        )
    )

    svc = RAGAnswerService(rag_service=mock_rag_svc, answer_generator=mock_answer_gen)
    request = RAGAnswerRequest(query="Explain system architecture", top_k=5)

    resp: RAGAnswerResponse = await svc.answer_question(
        collection_id=col_uuid,
        request=request,
        db=db_session,
    )

    assert resp.execution_trace is not None
    assert len(resp.execution_trace.stages) == 9
    stage_ids = [s.id for s in resp.execution_trace.stages]
    assert stage_ids == [
        "stage-1-query",
        "stage-2-validation",
        "stage-3-collection",
        "stage-4-retrieval",
        "stage-5-filtering",
        "stage-6-context",
        "stage-7-generation",
        "stage-8-citation",
        "stage-9-final",
    ]
    # Check all stages completed and timing accuracy rules
    for stage in resp.execution_trace.stages:
        assert stage.status == "completed"
        assert stage.started_at is not None
        if stage.id in ("stage-4-retrieval", "stage-5-filtering", "stage-6-context"):
            assert stage.duration_ms is None
        else:
            assert stage.duration_ms is not None
    assert resp.execution_trace.total_duration_ms > 0.0


@pytest.mark.asyncio
async def test_rag_answer_service_execution_trace_insufficient_evidence(db_session: AsyncSession):
    col = await _create_test_collection(db_session)
    col_uuid = UUID(col.id)

    mock_rag_resp = RAGResponse(
        query="What is the undisclosed roadmap?",
        collection_id=col.id,
        retrieved_chunk_count=0,
        assembled_context="",
        context_character_count=0,
        sources=[],
        prompt_text="",
        excluded_chunks=[],
        retrieval_metadata={"candidate_count": 0},
    )

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(return_value=mock_rag_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="The provided documents do not contain sufficient information to answer this question.",
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            provider="rule_based",
            model="none",
            usage=None,
            latency_ms=0.0,
            citations=CitationValidationResult(),
            error_message=None,
        )
    )

    svc = RAGAnswerService(rag_service=mock_rag_svc, answer_generator=mock_answer_gen)
    request = RAGAnswerRequest(query="What is the undisclosed roadmap?")

    resp: RAGAnswerResponse = await svc.answer_question(
        collection_id=col_uuid,
        request=request,
        db=db_session,
    )

    assert resp.execution_trace is not None
    gen_stage = next(s for s in resp.execution_trace.stages if s.id == "stage-7-generation")
    cit_stage = next(s for s in resp.execution_trace.stages if s.id == "stage-8-citation")
    assert gen_stage.status == "skipped"
    assert cit_stage.status == "skipped"

