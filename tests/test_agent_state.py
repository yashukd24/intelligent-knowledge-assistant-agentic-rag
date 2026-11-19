"""
tests/test_agent_state.py
─────────────────────────────────────────────────────────────
Unit tests for Phase 10 AgentState and related schemas:
  - AgentState initialization and default values.
  - Pydantic v2 serialization / deserialization roundtrip.
  - AgentEvent immutability and frozen attribute enforcement.
  - AgentRunRequest and AgentRunResponse validation rules.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime
import json
from uuid import uuid4
import pytest
from pydantic import ValidationError

from backend.schemas.agent_state import (
    AgentEvent,
    AgentEventType,
    AgentRunRequest,
    AgentRunResponse,
    AgentState,
    AgentStatus,
)
from backend.schemas.rag import SourceReference
from backend.schemas.retrieval import RetrievedChunk


def test_agent_state_initialization():
    col_id = str(uuid4())
    state = AgentState(
        collection_id=col_id,
        original_query="What are the key findings?",
        normalized_query="What are the key findings?",
    )

    assert state.collection_id == col_id
    assert state.status == AgentStatus.INITIALIZING
    assert state.current_step == 0
    assert state.iteration_count == 0
    assert state.max_iterations == 5
    assert state.is_sufficient is False
    assert state.evidence_score == 0.0
    assert state.answer is None
    assert state.execution_plan == []
    assert state.selected_tools == []
    assert state.retrieved_chunks == []
    assert state.sources == []
    assert state.citations == []
    assert state.invalid_citations == []
    assert state.agent_events == []
    assert state.error_details is None
    assert isinstance(state.session_id, str)
    assert isinstance(state.trace_id, str)


def test_agent_state_serialization_roundtrip():
    col_id = str(uuid4())
    chunk = RetrievedChunk(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        collection_id=col_id,
        text="Sample retrieved content for testing.",
        similarity_score=0.88,
        distance=0.12,
        chunk_index=0,
        page_number=1,
    )
    src = SourceReference(
        source_id="[S1]",
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        filename="test.pdf",
        page_number=1,
        chunk_index=0,
        similarity_score=0.88,
        char_count=37,
    )
    event = AgentEvent(
        event_type=AgentEventType.QUERY_UNDERSTOOD,
        step=1,
        payload={"query": "What is AI?"},
        message="Query understood successfully.",
    )

    state = AgentState(
        collection_id=col_id,
        original_query="What is AI?",
        normalized_query="What is AI?",
        status=AgentStatus.SYNTHESIZING,
        current_step=4,
        iteration_count=0,
        retrieved_chunks=[chunk],
        sources=[src],
        is_sufficient=True,
        evidence_score=0.88,
        answer="AI is artificial intelligence [S1].",
        citations=["[S1]"],
        agent_events=[event],
    )

    # Dump to JSON and re-parse
    json_str = state.model_dump_json()
    parsed_data = json.loads(json_str)
    assert parsed_data["status"] == "SYNTHESIZING"
    assert parsed_data["evidence_score"] == 0.88
    assert len(parsed_data["sources"]) == 1
    assert parsed_data["sources"][0]["source_id"] == "[S1]"

    # Reconstitute into model
    reconstituted = AgentState.model_validate_json(json_str)
    assert reconstituted.session_id == state.session_id
    assert reconstituted.status == AgentStatus.SYNTHESIZING
    assert len(reconstituted.agent_events) == 1
    assert reconstituted.agent_events[0].event_type == AgentEventType.QUERY_UNDERSTOOD
    assert reconstituted.citations == ["[S1]"]


def test_agent_event_frozen_immutability():
    event = AgentEvent(
        event_type=AgentEventType.PLAN_CREATED,
        step=1,
        payload={"plan": ["step1", "step2"]},
        message="Plan initialized",
    )

    with pytest.raises(ValidationError):
        # Attempt to mutate frozen event model attribute
        event.message = "Mutated message"  # type: ignore


def test_agent_run_request_validation():
    # 1. Valid request
    req = AgentRunRequest(query="Explain agent state machine", top_k=5)
    assert req.query == "Explain agent state machine"
    assert req.top_k == 5

    # 2. Empty query rejected
    with pytest.raises(ValidationError):
        AgentRunRequest(query="   ")

    # 3. Invalid top_k
    with pytest.raises(ValidationError):
        AgentRunRequest(query="Valid query", top_k=0)

    # 4. Invalid similarity threshold out of range
    with pytest.raises(ValidationError):
        AgentRunRequest(query="Valid query", similarity_threshold=2.5)

    # 5. Invalid max_context_chars below minimum
    with pytest.raises(ValidationError):
        AgentRunRequest(query="Valid query", max_context_chars=100)
