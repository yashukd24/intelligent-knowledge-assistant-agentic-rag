"""
tests/test_agent_controller.py
─────────────────────────────────────────────────────────────
Deterministic unit tests for AgentController state machine:
  - Transition guards and matrix enforcement.
  - Invalid transition rejections.
  - Terminal state immutability.
  - Successful baseline agent execution.
  - Insufficient-evidence rule-based refusal (zero LLM inference).
  - Provider failure and timeout handling.
  - Async cancellation preservation.
─────────────────────────────────────────────────────────────
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import pytest

from backend.schemas.agent_state import (
    AgentEventType,
    AgentRunRequest,
    AgentRunResponse,
    AgentState,
    AgentStatus,
)
from backend.schemas.rag import RAGResponse, SourceReference
from backend.schemas.retrieval import RetrievedChunk
from backend.services.agent_controller import (
    AgentController,
    InvalidStateTransitionError,
)
from backend.services.answer_generator import (
    AnswerGenerationResult,
    AnswerGenerator,
    AnswerStatus,
    STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
)
from backend.services.citation_validator import CitationValidationResult, CitationValidator
from backend.services.context_builder import ContextBuilder
from backend.services.llm_provider import TokenUsage
from backend.services.prompt_builder import PromptBuilder
from backend.services.rag_service import RAGService
from backend.services.retrieval_service import RetrievalService


def _create_mock_source(col_id: str, similarity_score: float = 0.85) -> SourceReference:
    return SourceReference(
        source_id="[S1]",
        document_id=str(uuid4()),
        chunk_id=str(uuid4()),
        filename="benchmark.pdf",
        page_number=1,
        chunk_index=0,
        similarity_score=similarity_score,
        char_count=50,
    )


# ── Transition Matrix & Guard Tests ──────────────────────────────────────────


def test_transition_guards_valid_sequence():
    controller = AgentController()
    state = AgentState(
        collection_id=str(uuid4()),
        original_query="What is agentic RAG?",
        normalized_query="What is agentic RAG?",
    )
    assert state.status == AgentStatus.INITIALIZING

    # Step 1: INITIALIZING -> PLANNING
    controller.transition_to(
        state,
        AgentStatus.PLANNING,
        AgentEventType.PLAN_CREATED,
        "Plan generated",
    )
    assert state.status == AgentStatus.PLANNING
    assert state.current_step == 1

    # Step 2: PLANNING -> RETRIEVING
    controller.transition_to(
        state,
        AgentStatus.RETRIEVING,
        AgentEventType.TOOL_SELECTED,
        "Retrieval tool chosen",
    )
    assert state.status == AgentStatus.RETRIEVING
    assert state.current_step == 2

    # Step 3: RETRIEVING -> EVALUATING
    controller.transition_to(
        state,
        AgentStatus.EVALUATING,
        AgentEventType.EVIDENCE_EVALUATED,
        "Evidence score computed",
    )
    assert state.status == AgentStatus.EVALUATING
    assert state.current_step == 3

    # Step 4: EVALUATING -> SYNTHESIZING
    controller.transition_to(
        state,
        AgentStatus.SYNTHESIZING,
        AgentEventType.ANSWER_SYNTHESIZED,
        "Drafting answer",
    )
    assert state.status == AgentStatus.SYNTHESIZING
    assert state.current_step == 4

    # Step 5: SYNTHESIZING -> COMPLETED
    controller.transition_to(
        state,
        AgentStatus.COMPLETED,
        AgentEventType.AGENT_DONE,
        "Done",
    )
    assert state.status == AgentStatus.COMPLETED
    assert state.current_step == 5


def test_transition_guards_invalid_transition_raises():
    controller = AgentController()
    state = AgentState(
        collection_id=str(uuid4()),
        original_query="Illegal skip check",
        normalized_query="Illegal skip check",
    )

    # Cannot jump straight from INITIALIZING to SYNTHESIZING
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        controller.transition_to(
            state,
            AgentStatus.SYNTHESIZING,
            AgentEventType.ANSWER_SYNTHESIZED,
            "Illegal skip",
        )
    assert "Illegal state transition from INITIALIZING to SYNTHESIZING" in str(exc_info.value)


def test_terminal_state_protection():
    controller = AgentController()
    state = AgentState(
        collection_id=str(uuid4()),
        original_query="Terminal check",
        normalized_query="Terminal check",
        status=AgentStatus.COMPLETED,
    )

    # Terminal COMPLETED cannot transition to any state
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        controller.transition_to(
            state,
            AgentStatus.PLANNING,
            AgentEventType.PLAN_CREATED,
            "Illegal re-open",
        )
    assert "Illegal state transition from COMPLETED to PLANNING" in str(exc_info.value)


# ── Execution Lifecycle Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_controller_successful_execution():
    col_id = str(uuid4())
    src = _create_mock_source(col_id, similarity_score=0.92)

    mock_rag_resp = RAGResponse(
        query="What is reinforcement learning?",
        collection_id=col_id,
        retrieved_chunk_count=1,
        assembled_context="[S1]: Reinforcement learning trains agents via rewards.",
        context_character_count=52,
        sources=[src],
        prompt_text="Context: [S1] RL. Question: What is RL?",
        excluded_chunks=[],
        retrieval_metadata={"candidate_count": 1},
    )

    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(return_value=mock_rag_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="Reinforcement learning uses reward signals [S1].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            latency_ms=250.0,
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

    controller = AgentController(
        rag_service=mock_rag_svc,
        answer_generator=mock_answer_gen,
    )

    request = AgentRunRequest(query="What is reinforcement learning?", top_k=3)
    response = await controller.run(request=request, collection_id=uuid4())

    assert response.status == AgentStatus.COMPLETED
    assert response.is_sufficient is True
    assert response.evidence_score == 0.92
    assert response.answer == "Reinforcement learning uses reward signals [S1]."
    assert response.valid_citations == ["[S1]"]
    assert len(response.events) >= 6
    assert response.execution_trace is not None
    assert len(response.execution_trace.stages) >= 6


@pytest.mark.asyncio
async def test_agent_controller_insufficient_evidence_bypasses_llm():
    col_id = str(uuid4())

    # Empty retrieved sources
    mock_rag_resp = RAGResponse(
        query="Unrelated query?",
        collection_id=col_id,
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
    mock_answer_gen.generate_answer = AsyncMock()

    controller = AgentController(
        rag_service=mock_rag_svc,
        answer_generator=mock_answer_gen,
    )

    request = AgentRunRequest(query="Unrelated query?", top_k=5)
    response = await controller.run(request=request, collection_id=uuid4())

    assert response.status == AgentStatus.COMPLETED
    assert response.is_sufficient is False
    assert response.evidence_score == 0.0
    assert response.answer == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
    # CRITICAL: LLM generate_answer must NEVER be invoked for insufficient evidence
    mock_answer_gen.generate_answer.assert_not_called()


@pytest.mark.asyncio
async def test_agent_controller_provider_failure_handling():
    col_id = str(uuid4())
    src = _create_mock_source(col_id)

    mock_rag_resp = RAGResponse(
        query="Test query",
        collection_id=col_id,
        retrieved_chunk_count=1,
        assembled_context="[S1]: Context",
        context_character_count=13,
        sources=[src],
        prompt_text="Prompt",
        excluded_chunks=[],
        retrieval_metadata={"candidate_count": 1},
    )
    mock_rag_svc = MagicMock(spec=RAGService)
    mock_rag_svc.build_rag_context = AsyncMock(return_value=mock_rag_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="",
            status=AnswerStatus.PROVIDER_ERROR,
            provider="ollama",
            model="qwen2.5:3b",
            latency_ms=10.0,
            citations=CitationValidationResult(),
            error_message="Ollama connection refused",
        )
    )

    controller = AgentController(
        rag_service=mock_rag_svc,
        answer_generator=mock_answer_gen,
    )

    request = AgentRunRequest(query="Test query")
    response = await controller.run(request=request, collection_id=uuid4())

    assert response.status == AgentStatus.FAILED
    assert "Ollama connection refused" in (response.error_details or "")


@pytest.mark.asyncio
async def test_agent_controller_async_cancellation_preservation():
    mock_rag_svc = MagicMock(spec=RAGService)
    # Simulate an external task cancellation during retrieval
    mock_rag_svc.build_rag_context = AsyncMock(side_effect=asyncio.CancelledError())

    controller = AgentController(rag_service=mock_rag_svc)
    request = AgentRunRequest(query="Cancel test")

    with pytest.raises(asyncio.CancelledError):
        await controller.run(request=request, collection_id=uuid4())


@pytest.mark.asyncio
async def test_agent_controller_multi_query_decomposition_and_deduplication():
    col_id = str(uuid4())
    doc_id = str(uuid4())

    chunk_a = RetrievedChunk(
        chunk_id="chunk-A",
        document_id=doc_id,
        collection_id=col_id,
        text="Operating revenue in Q3 2025 reached $4.2B.",
        similarity_score=0.85,
        distance=0.15,
        chunk_index=0,
    )
    # chunk_b appears in subquery 1 with 0.70 and in subquery 2 with 0.92
    chunk_b_low = RetrievedChunk(
        chunk_id="chunk-B",
        document_id=doc_id,
        collection_id=col_id,
        text="Capital expenditures and operational revenue summary.",
        similarity_score=0.70,
        distance=0.30,
        chunk_index=1,
    )
    chunk_b_high = RetrievedChunk(
        chunk_id="chunk-B",
        document_id=doc_id,
        collection_id=col_id,
        text="Capital expenditures and operational revenue summary.",
        similarity_score=0.92,
        distance=0.08,
        chunk_index=1,
    )
    chunk_c = RetrievedChunk(
        chunk_id="chunk-C",
        document_id=doc_id,
        collection_id=col_id,
        text="Capital expenditures in Q3 were $600M.",
        similarity_score=0.88,
        distance=0.12,
        chunk_index=2,
    )

    resp_sq1 = MagicMock()
    resp_sq1.chunks = [chunk_a, chunk_b_low]

    resp_sq2 = MagicMock()
    resp_sq2.chunks = [chunk_b_high, chunk_c]

    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(side_effect=[resp_sq1, resp_sq2])

    real_rag_svc = RAGService(
        retrieval_service=mock_retrieval_svc,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="Operating revenue was $4.2B [S2] and CapEx was $600M [S3].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet",
            latency_ms=300.0,
            citations=CitationValidationResult(
                cited_sources=["[S2]", "[S3]"],
                valid_citations=["[S2]", "[S3]"],
            ),
        )
    )

    controller = AgentController(
        rag_service=real_rag_svc,
        answer_generator=mock_answer_gen,
    )

    compound_query = "What was the operating revenue in Q3 2025, and what were the major capital expenditures?"
    request = AgentRunRequest(query=compound_query, top_k=2)
    response = await controller.run(request=request, collection_id=uuid4())

    assert response.status == AgentStatus.COMPLETED
    assert response.is_sufficient is True
    # Verify subqueries were derived
    assert len(response.decomposed_queries) == 2
    assert response.structured_plan is not None
    assert response.structured_plan.is_decomposed is True
    assert len(response.structured_plan.steps) == 5

    # Verify deduplication: 3 unique chunks instead of 4
    assert len(response.sources) == 3
    # Verify highest similarity score retained for chunk-B (0.92)
    chunk_b_source = next(s for s in response.sources if s.chunk_id == "chunk-B")
    assert chunk_b_source.similarity_score == 0.92

    # Verify event stream: QUERY_UNDERSTOOD -> QUERY_DECOMPOSED -> PLAN_CREATED
    event_types = [e.event_type for e in response.events]
    assert event_types[0] == AgentEventType.QUERY_UNDERSTOOD
    assert event_types[1] == AgentEventType.QUERY_DECOMPOSED
    assert event_types[2] == AgentEventType.PLAN_CREATED
