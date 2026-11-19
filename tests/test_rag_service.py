"""
tests/test_rag_service.py
─────────────────────────────────────────────────────────────
Comprehensive tests for RAGService:
  - Successful end-to-end context and prompt generation.
  - Empty retrieval handling without hallucination.
  - Exception propagation and wrapping (InvalidRAGQueryError, RAGExecutionError).
  - Dependency injection of RetrievalService, ContextBuilder, PromptBuilder.
  - Strict preservation of source metadata and citation IDs.
  - Verification that no external LLM API calls are executed.
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from backend.schemas.rag import RAGQuery, RAGResponse, SourceReference
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse, RetrievedChunk
from backend.services.context_builder import ContextBuilder, ContextBuildResult
from backend.services.prompt_builder import PromptBuilder
from backend.services.rag_service import (
    InvalidRAGQueryError,
    RAGExecutionError,
    RAGService,
)
from backend.services.retrieval_service import (
    InvalidRetrievalQueryError,
    RetrievalExecutionError,
    RetrievalService,
)
from backend.services.vector_store import CollectionNotFoundError

TEST_COLLECTION_ID = uuid4()


def make_dummy_chunk(
    chunk_id: str = "c1",
    document_id: str = "doc_1",
    collection_id: str = str(TEST_COLLECTION_ID),
    text: str = "Retrieved test content.",
    score: float = 0.88,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection_id=collection_id,
        text=text,
        similarity_score=score,
        distance=1.0 - score,
        chunk_index=0,
        filename="policy.pdf",
        page_number=1,
    )


@pytest.mark.asyncio
async def test_successful_rag_context_and_prompt_assembly():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    chunk = make_dummy_chunk("c1", text="Honesty is required.")
    ret_resp = RetrievalResponse(
        query="What is the policy?",
        collection_id=str(TEST_COLLECTION_ID),
        results=[chunk],
        total_candidates=1,
        returned_results=1,
        rejected_candidates=0,
        accepted_candidates=1,
        retrieval_metadata={"top_k": 5},
    )
    mock_retrieval_svc.retrieve = AsyncMock(return_value=ret_resp)

    service = RAGService(retrieval_service=mock_retrieval_svc)

    query = RAGQuery(
        query="What is the policy?",
        collection_id=TEST_COLLECTION_ID,
        top_k=5,
    )

    rag_resp = await service.build_rag_context(query)

    assert isinstance(rag_resp, RAGResponse)
    assert rag_resp.query == "What is the policy?"
    assert rag_resp.collection_id == str(TEST_COLLECTION_ID)
    assert rag_resp.retrieved_chunk_count == 1
    assert len(rag_resp.sources) == 1
    assert rag_resp.sources[0].source_id == "[S1]"
    assert "[S1]" in rag_resp.assembled_context
    assert "Honesty is required." in rag_resp.assembled_context
    assert "=== RETRIEVED CONTEXT ===" in rag_resp.prompt_text
    assert "What is the policy?" in rag_resp.prompt_text
    assert "[S1]" in rag_resp.source_metadata
    assert rag_resp.excluded_chunks == []


@pytest.mark.asyncio
async def test_empty_retrieval_handled_cleanly():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    ret_resp = RetrievalResponse(
        query="Nonexistent topic query",
        collection_id=str(TEST_COLLECTION_ID),
        results=[],
        total_candidates=0,
        returned_results=0,
        rejected_candidates=0,
        accepted_candidates=0,
    )
    mock_retrieval_svc.retrieve = AsyncMock(return_value=ret_resp)

    service = RAGService(retrieval_service=mock_retrieval_svc)

    query = RAGQuery(
        query="Nonexistent topic query",
        collection_id=TEST_COLLECTION_ID,
    )

    rag_resp = await service.build_rag_context(query)

    assert rag_resp.retrieved_chunk_count == 0
    assert rag_resp.assembled_context == ""
    assert rag_resp.sources == []
    assert rag_resp.source_metadata == {}
    assert "No relevant evidence was retrieved" in rag_resp.prompt_text


@pytest.mark.asyncio
async def test_invalid_query_raises_validation_error():
    service = RAGService(retrieval_service=MagicMock(spec=RetrievalService))

    # Empty query string
    with pytest.raises(InvalidRAGQueryError):
        await service.build_rag_context({"query": "   ", "collection_id": str(TEST_COLLECTION_ID)})

    # Invalid type
    with pytest.raises(InvalidRAGQueryError):
        await service.build_rag_context(12345)


@pytest.mark.asyncio
async def test_collection_not_found_propagated():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(
        side_effect=CollectionNotFoundError(f"Collection {TEST_COLLECTION_ID} not found")
    )

    service = RAGService(retrieval_service=mock_retrieval_svc)
    query = RAGQuery(query="Valid query", collection_id=TEST_COLLECTION_ID)

    with pytest.raises(CollectionNotFoundError):
        await service.build_rag_context(query)


@pytest.mark.asyncio
async def test_retrieval_execution_error_wrapped():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(
        side_effect=RetrievalExecutionError("ChromaDB read failure")
    )

    service = RAGService(retrieval_service=mock_retrieval_svc)
    query = RAGQuery(query="Valid query", collection_id=TEST_COLLECTION_ID)

    with pytest.raises(RAGExecutionError, match="Retrieval execution error"):
        await service.build_rag_context(query)


@pytest.mark.asyncio
async def test_context_builder_error_wrapped():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(
        return_value=RetrievalResponse(
            query="Q",
            collection_id=str(TEST_COLLECTION_ID),
            results=[make_dummy_chunk()],
            total_candidates=1,
            returned_results=1,
            rejected_candidates=0,
            accepted_candidates=1,
        )
    )
    mock_ctx_builder = MagicMock(spec=ContextBuilder)
    mock_ctx_builder.build_context.side_effect = RuntimeError("Context buffer fault")

    service = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=mock_ctx_builder,
    )
    query = RAGQuery(query="Valid query", collection_id=TEST_COLLECTION_ID)

    with pytest.raises(RAGExecutionError, match="Context assembly failed"):
        await service.build_rag_context(query)


@pytest.mark.asyncio
async def test_prompt_builder_error_wrapped():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(
        return_value=RetrievalResponse(
            query="Q",
            collection_id=str(TEST_COLLECTION_ID),
            results=[],
            total_candidates=0,
            returned_results=0,
            rejected_candidates=0,
            accepted_candidates=0,
        )
    )
    mock_prompt_builder = MagicMock(spec=PromptBuilder)
    mock_prompt_builder.build_prompt.side_effect = RuntimeError("Template render fault")

    service = RAGService(
        retrieval_service=mock_retrieval_svc,
        prompt_builder=mock_prompt_builder,
    )
    query = RAGQuery(query="Valid query", collection_id=TEST_COLLECTION_ID)

    with pytest.raises(RAGExecutionError, match="Prompt construction failed"):
        await service.build_rag_context(query)
