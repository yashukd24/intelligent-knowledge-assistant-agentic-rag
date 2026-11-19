"""
tests/test_tool_registry.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for Phase 12 Tool Registry & Selection:
  1. Tool registration, discovery, and schema introspection.
  2. Input and output schema validation and extra-field rejection.
  3. Strict AST calculator security:
     - Allowlisted arithmetic operations (+, -, *, /, %, **).
     - Division by zero handling.
     - Rejection of variables, function calls, attributes, comprehensions.
     - Exponent and magnitude bounds protection.
  4. Tool execution, timeout handling, and exception containment.
  5. Central call budget enforcement (max 10 tool calls per run).
  6. Deterministic tool selection rules and evidence primacy.
  7. AgentController integration with ToolCall/ToolResult tracking.
─────────────────────────────────────────────────────────────
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from backend.schemas.agent_state import AgentEventType, AgentRunRequest, AgentStatus
from backend.schemas.query_plan import PlanStep, PlanStepType, QueryIntent
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse, RetrievedChunk
from backend.schemas.tool import (
    CalculatorInput,
    CalculatorOutput,
    MetadataFilterInput,
    MetadataFilterOutput,
    SemanticSearchInput,
    SemanticSearchOutput,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolStatus,
)
from backend.services.agent_controller import AgentController
from backend.services.answer_generator import (
    AnswerGenerationResult,
    AnswerGenerator,
    AnswerStatus,
)
from backend.services.citation_validator import CitationValidationResult
from backend.services.retrieval_service import RetrievalService
from backend.services.tool_registry import (
    BaseTool,
    CalculatorTool,
    MAX_TOOL_CALLS_PER_RUN,
    MetadataFilterTool,
    SafeCalculatorEvaluator,
    SemanticSearchTool,
    ToolRegistry,
)


# ── Fixtures & Helpers ───────────────────────────────────────────────────────


def _create_dummy_chunk(chunk_id: str, col_id: str, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(uuid4()),
        collection_id=col_id,
        text="Sample document chunk text for testing.",
        similarity_score=score,
        distance=round(1.0 - score, 4),
        chunk_index=0,
    )


# ── 1. Registry Registration & Discovery ─────────────────────────────────────


def test_registry_registration_and_discovery():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    sem_tool = SemanticSearchTool(retrieval_service=mock_retrieval_svc)
    meta_tool = MetadataFilterTool(retrieval_service=mock_retrieval_svc)
    calc_tool = CalculatorTool()

    registry = ToolRegistry(tools=[sem_tool, meta_tool, calc_tool])

    assert registry.has_tool("semantic_search")
    assert registry.has_tool("metadata_filter")
    assert registry.has_tool("calculator")
    assert not registry.has_tool("unregistered_tool")

    tools_list = registry.list_tools()
    assert len(tools_list) == 3

    tool_names = [t.name for t in tools_list]
    assert "semantic_search" in tool_names
    assert "metadata_filter" in tool_names
    assert "calculator" in tool_names

    # Verify ToolDefinition parameters introspection
    calc_def = next(t for t in tools_list if t.name == "calculator")
    assert len(calc_def.parameters) == 1
    assert calc_def.parameters[0].name == "expression"
    assert calc_def.parameters[0].type == "string"
    assert calc_def.is_safe is True


def test_empty_tool_name_rejection():
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    tool = SemanticSearchTool(retrieval_service=mock_retrieval_svc)
    tool.name = "   "
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Tool name cannot be empty"):
        registry.register(tool)


# ── 2. Input/Output Schema Validation ────────────────────────────────────────


def test_semantic_search_schema_validation():
    col_id = uuid4()
    # Valid input
    valid_input = SemanticSearchInput(
        query="What is reinforcement learning?",
        collection_id=col_id,
        top_k=5,
        similarity_threshold=0.75,
    )
    assert valid_input.top_k == 5

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        SemanticSearchInput(
            query="Test",
            collection_id=col_id,
            arbitrary_field="not_allowed",
        )

    # Empty query forbidden
    with pytest.raises(ValidationError):
        SemanticSearchInput(query="", collection_id=col_id)


def test_metadata_filter_schema_validation():
    col_id = uuid4()
    doc_id = uuid4()
    # Valid input
    valid_input = MetadataFilterInput(
        collection_id=col_id,
        document_ids=[doc_id],
        query="Filter query",
    )
    assert valid_input.document_ids == [doc_id]

    # Empty document_ids rejected
    with pytest.raises(ValidationError):
        MetadataFilterInput(collection_id=col_id, document_ids=[])


def test_calculator_schema_validation():
    # Valid input
    valid_input = CalculatorInput(expression="2500 * 1.15")
    assert valid_input.expression == "2500 * 1.15"

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        CalculatorInput(expression="1 + 1", extra_field=123)

    # Max length > 200 rejected
    with pytest.raises(ValidationError):
        CalculatorInput(expression="1 + " * 80)


def test_tool_call_and_result_immutability():
    call = ToolCall(tool_name="calculator", arguments={"expression": "1 + 1"})
    result = ToolResult(
        call_id=call.call_id,
        tool_name="calculator",
        status=ToolStatus.SUCCESS,
        output=CalculatorOutput(result=2.0, expression="1 + 1", rounded_result=2.0),
    )
    with pytest.raises(ValidationError):
        call.tool_name = "tampered"  # type: ignore
    with pytest.raises(ValidationError):
        result.status = ToolStatus.EXECUTION_ERROR  # type: ignore


# ── 3. Strict AST Calculator Security Tests ──────────────────────────────────


def test_calculator_safe_arithmetic():
    assert SafeCalculatorEvaluator.evaluate("10 + 20") == 30.0
    assert SafeCalculatorEvaluator.evaluate("100 - 45.5") == 54.5
    assert SafeCalculatorEvaluator.evaluate("12 * 8") == 96.0
    assert SafeCalculatorEvaluator.evaluate("100 / 4") == 25.0
    assert SafeCalculatorEvaluator.evaluate("17 % 5") == 2.0
    assert SafeCalculatorEvaluator.evaluate("2 ** 8") == 256.0
    assert SafeCalculatorEvaluator.evaluate("-(10 + 5) * 2") == -30.0
    assert SafeCalculatorEvaluator.evaluate("((4.5 * 2) + (10 / 2)) - 1") == 13.0


def test_calculator_division_by_zero():
    with pytest.raises(ZeroDivisionError, match="Division by zero"):
        SafeCalculatorEvaluator.evaluate("100 / 0")

    with pytest.raises(ZeroDivisionError, match="Modulo by zero"):
        SafeCalculatorEvaluator.evaluate("100 % 0")


@pytest.mark.parametrize(
    "malicious_expr",
    [
        "__import__('os').system('dir')",
        "eval('1 + 1')",
        "exec('x = 1')",
        "open('pyproject.toml').read()",
        "subprocess.run(['dir'])",
        "(lambda: 42)()",
        "[x for x in range(10)]",
        "x + 1",
        "math.sqrt(16)",
        "(1).__class__.__bases__",
        "{'a': 1}",
    ],
)
def test_calculator_rejects_unsafe_ast_constructs(malicious_expr: str):
    with pytest.raises(ValueError) as exc_info:
        SafeCalculatorEvaluator.evaluate(malicious_expr)
    assert any(
        msg in str(exc_info.value)
        for msg in [
            "Disallowed syntax node",
            "Invalid arithmetic syntax",
            "Unsupported unary operator",
            "Unsupported binary operator",
        ]
    )


def test_calculator_exponent_and_magnitude_bounds():
    # Exponent > 100 rejected
    with pytest.raises(ValueError, match="Exponent 101.0 exceeds safe limit"):
        SafeCalculatorEvaluator.evaluate("2 ** 101")

    # Exponent with huge base rejected
    with pytest.raises(ValueError, match="exceeds safe computation limits"):
        SafeCalculatorEvaluator.evaluate("10000000 ** 5")

    # Magnitude limit exceeded
    with pytest.raises(ValueError, match="exceeds magnitude limit"):
        SafeCalculatorEvaluator.evaluate("1e16 + 1e16")


# ── 4. Tool Execution, Timeout, and Exception Containment ────────────────────


@pytest.mark.asyncio
async def test_tool_execution_success():
    calc = CalculatorTool()
    registry = ToolRegistry(tools=[calc])

    result = await registry.execute_tool("calculator", {"expression": "25 * 4"})
    assert result.status == ToolStatus.SUCCESS
    assert isinstance(result.output, CalculatorOutput)
    assert result.output.result == 100.0
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_tool_not_found():
    registry = ToolRegistry()
    result = await registry.execute_tool("missing_tool", {})
    assert result.status == ToolStatus.TOOL_NOT_FOUND
    assert "is not registered" in (result.error_message or "")


@pytest.mark.asyncio
async def test_tool_invalid_arguments():
    calc = CalculatorTool()
    registry = ToolRegistry(tools=[calc])

    # Missing required argument 'expression'
    result = await registry.execute_tool("calculator", {})
    assert result.status == ToolStatus.INVALID_ARGUMENTS
    assert "validation error" in (result.error_message or "").lower()


@pytest.mark.asyncio
async def test_tool_execution_timeout():
    class SlowTool(BaseTool):
        name = "slow_tool"
        description = "A deliberately slow tool for timeout verification."
        input_schema = CalculatorInput
        output_schema = CalculatorOutput
        timeout_seconds = 0.05

        async def execute(self, validated_input: BaseModel, context=None) -> BaseModel:
            await asyncio.sleep(0.5)
            return CalculatorOutput(result=1.0, expression="1", rounded_result=1.0)

    slow_tool = SlowTool()
    registry = ToolRegistry(tools=[slow_tool])

    result = await registry.execute_tool("slow_tool", {"expression": "1 + 1"})
    assert result.status == ToolStatus.TIMEOUT
    assert "timed out after 0.05s" in (result.error_message or "")


@pytest.mark.asyncio
async def test_tool_exception_containment():
    class FailingTool(BaseTool):
        name = "failing_tool"
        description = "A tool that raises an internal exception."
        input_schema = CalculatorInput
        output_schema = CalculatorOutput

        async def execute(self, validated_input: BaseModel, context=None) -> BaseModel:
            raise RuntimeError("Database connection pool exhausted.")

    failing_tool = FailingTool()
    registry = ToolRegistry(tools=[failing_tool])

    result = await registry.execute_tool("failing_tool", {"expression": "1 + 1"})
    assert result.status == ToolStatus.EXECUTION_ERROR
    assert "Database connection pool exhausted" in (result.error_message or "")


# ── 5. Central Call Budget Enforcement (Max 10 Calls) ────────────────────────


@pytest.mark.asyncio
async def test_central_tool_call_budget_exceeded():
    calc = CalculatorTool()
    registry = ToolRegistry(tools=[calc])

    # Call with call_count = 10 (exceeding maximum permitted 10 calls)
    result = await registry.execute_tool(
        "calculator",
        {"expression": "1 + 1"},
        call_count=MAX_TOOL_CALLS_PER_RUN,
    )
    assert result.status == ToolStatus.CALL_LIMIT_EXCEEDED
    assert "tool call limit of 10 exceeded" in (result.error_message or "")


# ── 6. Deterministic Tool Selection Rules & Evidence Primacy ─────────────────


def test_tool_selection_retrieval_primacy():
    mock_retrieval = MagicMock(spec=RetrievalService)
    registry = ToolRegistry(
        tools=[
            SemanticSearchTool(retrieval_service=mock_retrieval),
            MetadataFilterTool(retrieval_service=mock_retrieval),
            CalculatorTool(),
        ]
    )

    retrieval_step = PlanStep(
        step_number=1,
        step_type=PlanStepType.RETRIEVAL,
        name="Targeted Semantic Retrieval",
        description="Execute vector search",
    )

    # Retrieval step without document filter selects semantic_search
    selected = registry.select_tool(step=retrieval_step, has_document_filter=False)
    assert selected == "semantic_search"

    # Retrieval step with document filter selects metadata_filter
    selected_filtered = registry.select_tool(step=retrieval_step, has_document_filter=True)
    assert selected_filtered == "metadata_filter"

    # CRITICAL: Even if query mentions numbers, retrieval step MUST NOT select calculator!
    selected_math_query = registry.select_tool(
        step=retrieval_step,
        query_intent=QueryIntent.FACTUAL,
        subquery_text="Calculate the 2025 operating profit",
        has_document_filter=False,
    )
    assert selected_math_query == "semantic_search"


def test_tool_selection_dedicated_calculation():
    registry = ToolRegistry(tools=[CalculatorTool()])

    # Explicit arithmetic question without retrieval step selects calculator
    selected = registry.select_tool(
        step=None,
        query_intent=QueryIntent.FACTUAL,
        subquery_text="calculate 500 * 1.25",
        has_document_filter=False,
    )
    assert selected == "calculator"


# ── 7. AgentController Integration with Tool Audit State ─────────────────────


@pytest.mark.asyncio
async def test_agent_controller_tool_tracking_in_state():
    col_id = uuid4()
    mock_retrieval_resp = RetrievalResponse(
        query="What is reinforcement learning?",
        collection_id=str(col_id),
        results=[_create_dummy_chunk("c1", str(col_id), score=0.91)],
        total_candidates=1,
        returned_results=1,
    )
    mock_retrieval_svc = MagicMock(spec=RetrievalService)
    mock_retrieval_svc.retrieve = AsyncMock(return_value=mock_retrieval_resp)

    mock_answer_gen = MagicMock(spec=AnswerGenerator)
    mock_answer_gen.generate_answer = AsyncMock(
        return_value=AnswerGenerationResult(
            answer="Reinforcement learning optimizes actions [S1].",
            status=AnswerStatus.ANSWERED,
            provider="anthropic",
            model="claude-3-5-sonnet",
            citations=CitationValidationResult(valid_citations=["[S1]"]),
        )
    )

    sem_tool = SemanticSearchTool(retrieval_service=mock_retrieval_svc)
    registry = ToolRegistry(tools=[sem_tool])

    controller = AgentController(
        answer_generator=mock_answer_gen,
        tool_registry=registry,
    )

    request = AgentRunRequest(query="What is reinforcement learning?", top_k=2)
    response = await controller.run(request=request, collection_id=col_id)

    assert response.status == AgentStatus.COMPLETED
    assert response.is_sufficient is True

    # Verify tool calls and results recorded in AgentRunResponse
    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0].tool_name == "semantic_search"
    assert response.tool_calls[0].arguments["query"] == "What is reinforcement learning?"

    assert len(response.tool_results) >= 1
    assert response.tool_results[0].status == ToolStatus.SUCCESS
    assert response.tool_results[0].tool_name == "semantic_search"

    # Verify tool events in chronological log
    event_types = [e.event_type for e in response.events]
    assert AgentEventType.TOOL_SELECTED in event_types
    assert AgentEventType.TOOL_EXECUTED in event_types
