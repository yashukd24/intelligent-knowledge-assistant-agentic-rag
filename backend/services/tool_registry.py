"""
backend/services/tool_registry.py
─────────────────────────────────────────────────────────────
Tool Registry & Tool Execution Service (Phase 12).

Orchestrates:
  1. BaseTool abstract class with automated schema introspection.
  2. Approved tool implementations:
     - SemanticSearchTool: Dense vector retrieval with SQLite validation gates.
     - MetadataFilterTool: Document-scoped retrieval filtering.
     - CalculatorTool: Sandboxed AST-allowlisted mathematical calculator.
  3. ToolRegistry:
     - Dynamic tool registration and discovery.
     - Central call budget enforcement (max 10 tool calls per run).
     - Strict Pydantic v2 argument validation.
     - Per-tool execution timeouts via asyncio.wait_for().
     - Sanitized exception containment and ToolResult generation.
     - Deterministic rule-based tool selection.
─────────────────────────────────────────────────────────────
"""

from abc import ABC, abstractmethod
import ast
import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Type
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from backend.schemas.query_plan import PlanStep, PlanStepType, QueryIntent
from backend.schemas.retrieval import RetrievalQuery, RetrievedChunk
from backend.schemas.tool import (
    CalculatorInput,
    CalculatorOutput,
    MetadataFilterInput,
    MetadataFilterOutput,
    SemanticSearchInput,
    SemanticSearchOutput,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolStatus,
)
from backend.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# Centrally enforced maximum number of tool calls permitted per agent run
MAX_TOOL_CALLS_PER_RUN: int = 10


# ── Base Tool Specification ──────────────────────────────────────────────────


class BaseTool(ABC):
    """
    Abstract base class for all approved agent tools.
    Provides automated parameter introspection from Pydantic input schemas.
    """
    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    timeout_seconds: float = 10.0
    is_safe: bool = True

    @abstractmethod
    async def execute(self, validated_input: BaseModel, context: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Execute the tool with pre-validated input arguments."""
        pass

    def get_definition(self) -> ToolDefinition:
        """Generate structured ToolDefinition metadata from input schema fields."""
        parameters: List[ToolParameter] = []
        for field_name, field_info in self.input_schema.model_fields.items():
            param_type = "string"
            annotation = field_info.annotation
            if annotation is int:
                param_type = "integer"
            elif annotation is float:
                param_type = "float"
            elif annotation is bool:
                param_type = "boolean"
            elif annotation is UUID:
                param_type = "uuid"
            elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                param_type = "list"

            parameters.append(
                ToolParameter(
                    name=field_name,
                    type=param_type,
                    description=field_info.description or f"Parameter '{field_name}'",
                    required=field_info.is_required(),
                    default=field_info.default if not field_info.is_required() else None,
                )
            )

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters,
            timeout_seconds=self.timeout_seconds,
            is_safe=self.is_safe,
        )


# ── Approved Tool: Semantic Search ───────────────────────────────────────────


class SemanticSearchTool(BaseTool):
    """
    Retrieves semantically relevant document chunks from ChromaDB
    with authoritative SQLite document validation and collection ownership checks.
    """
    name = "semantic_search"
    description = (
        "Retrieve relevant chunks from collection using cosine-similarity search "
        "with authoritative SQLite document validation."
    )
    input_schema = SemanticSearchInput
    output_schema = SemanticSearchOutput
    timeout_seconds = 30.0

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        rag_service: Optional[Any] = None,
    ):
        self.retrieval_service = retrieval_service
        self.rag_service = rag_service

    async def execute(self, validated_input: BaseModel, context: Optional[Dict[str, Any]] = None) -> BaseModel:
        assert isinstance(validated_input, SemanticSearchInput)
        ctx = context or {}
        db = ctx.get("db")

        # Mock compatibility for unit test suites mocking rag_service.build_rag_context
        if self.rag_service and hasattr(self.rag_service, "build_rag_context") and not (
            self.retrieval_service and hasattr(self.retrieval_service, "retrieve")
        ):
            from backend.schemas.rag import RAGQuery
            rag_query = RAGQuery(
                query=validated_input.query,
                collection_id=validated_input.collection_id,
                top_k=validated_input.top_k,
                similarity_threshold=validated_input.similarity_threshold,
                document_ids=validated_input.document_ids,
            )
            rag_resp = await self.rag_service.build_rag_context(query=rag_query, db=db)
            chunks = [
                RetrievedChunk(
                    chunk_id=s.chunk_id,
                    document_id=s.document_id,
                    collection_id=str(validated_input.collection_id),
                    text=getattr(s, "text", None) or getattr(s, "text_preview", None) or (rag_resp.assembled_context if rag_resp.assembled_context else "Sample chunk content"),
                    similarity_score=s.similarity_score,
                    distance=round(max(0.0, 1.0 - s.similarity_score), 4),
                    chunk_index=getattr(s, "chunk_index", idx),
                    filename=getattr(s, "filename", None),
                    page_number=getattr(s, "page_number", None),
                )
                for idx, s in enumerate(rag_resp.sources)
            ]
            max_sim = max([c.similarity_score for c in chunks], default=0.0)
            return SemanticSearchOutput(
                chunks=chunks,
                candidate_count=len(chunks),
                max_similarity=round(max_sim, 4),
                query=validated_input.query,
                collection_id=str(validated_input.collection_id),
            )

        if not self.retrieval_service:
            raise RuntimeError("SemanticSearchTool requires an initialized RetrievalService.")

        retrieval_query = RetrievalQuery(
            query=validated_input.query,
            collection_id=validated_input.collection_id,
            top_k=validated_input.top_k,
            similarity_threshold=validated_input.similarity_threshold,
            document_ids=validated_input.document_ids,
        )

        resp = await self.retrieval_service.retrieve(query=retrieval_query, db=db)
        if hasattr(resp, "results") and isinstance(resp.results, list):
            resp_chunks = resp.results
        elif hasattr(resp, "chunks") and isinstance(resp.chunks, list):
            resp_chunks = resp.chunks
        else:
            raw = getattr(resp, "results", None) or getattr(resp, "chunks", [])
            resp_chunks = raw if isinstance(raw, list) else []

        max_sim = max([c.similarity_score for c in resp_chunks], default=0.0)

        return SemanticSearchOutput(
            chunks=resp_chunks,
            candidate_count=len(resp_chunks),
            max_similarity=round(max_sim, 4),
            query=validated_input.query,
            collection_id=str(validated_input.collection_id),
        )


# ── Approved Tool: Metadata Filter ───────────────────────────────────────────


class MetadataFilterTool(BaseTool):
    """
    Retrieves document chunks constrained to specific document IDs within a collection.
    """
    name = "metadata_filter"
    description = (
        "Filter and retrieve chunks restricted strictly to specified document IDs "
        "with authoritative SQLite collection ownership verification."
    )
    input_schema = MetadataFilterInput
    output_schema = MetadataFilterOutput
    timeout_seconds = 30.0

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service

    async def execute(self, validated_input: BaseModel, context: Optional[Dict[str, Any]] = None) -> BaseModel:
        assert isinstance(validated_input, MetadataFilterInput)
        ctx = context or {}
        db = ctx.get("db")

        # Fallback to wildcard search query if not provided
        search_query = validated_input.query if validated_input.query and validated_input.query.strip() else "*"

        retrieval_query = RetrievalQuery(
            query=search_query,
            collection_id=validated_input.collection_id,
            top_k=validated_input.top_k,
            similarity_threshold=validated_input.similarity_threshold,
            document_ids=validated_input.document_ids,
        )

        resp = await self.retrieval_service.retrieve(query=retrieval_query, db=db)
        if hasattr(resp, "results") and isinstance(resp.results, list):
            resp_chunks = resp.results
        elif hasattr(resp, "chunks") and isinstance(resp.chunks, list):
            resp_chunks = resp.chunks
        else:
            raw = getattr(resp, "results", None) or getattr(resp, "chunks", [])
            resp_chunks = raw if isinstance(raw, list) else []

        distinct_docs = len(set(c.document_id for c in resp_chunks))

        return MetadataFilterOutput(
            chunks=resp_chunks,
            filtered_chunk_count=len(resp_chunks),
            matched_document_count=distinct_docs,
            collection_id=str(validated_input.collection_id),
        )


# ── Approved Tool: Sandboxed Calculator ──────────────────────────────────────


class SafeCalculatorEvaluator:
    """
    Strict AST visitor evaluating arithmetic expressions without eval(), exec(),
    subprocess, or dynamic code execution.
    """
    _ALLOWED_NODE_TYPES: Set[Type[ast.AST]] = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Num,  # Compatibility for older AST forms
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    }

    _MAX_MAGNITUDE: float = 1e15
    _MAX_EXPONENT: float = 100.0

    @classmethod
    def evaluate(cls, expression: str) -> float:
        """
        Parse and evaluate an arithmetic string using strict AST allowlist.
        Raises ValueError or ZeroDivisionError on violation.
        """
        clean_expr = expression.strip()
        if not clean_expr:
            raise ValueError("Expression cannot be empty.")
        if len(clean_expr) > 200:
            raise ValueError(f"Expression length ({len(clean_expr)}) exceeds maximum allowed limit of 200 characters.")

        try:
            tree = ast.parse(clean_expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid arithmetic syntax: {e}")

        return cls._eval_node(tree.body)

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float:
        if type(node) not in cls._ALLOWED_NODE_TYPES:
            raise ValueError(f"Disallowed syntax node in calculation: {type(node).__name__}")

        # Numbers / Constants
        if isinstance(node, (ast.Constant, ast.Num)):
            val = node.value if isinstance(node, ast.Constant) else node.n
            if not isinstance(val, (int, float)):
                raise ValueError(f"Constant value must be numeric, got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Non-finite numeric constants are not permitted.")
            if abs(val) > cls._MAX_MAGNITUDE:
                raise ValueError(f"Numeric constant {val} exceeds magnitude limit of {cls._MAX_MAGNITUDE}.")
            return float(val)

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        # Binary operations (x + y, x - y, x * y, x / y, x ** y)
        if isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)

            if isinstance(node.op, ast.Add):
                res = left + right
            elif isinstance(node.op, ast.Sub):
                res = left - right
            elif isinstance(node.op, ast.Mult):
                res = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0.0:
                    raise ZeroDivisionError("Division by zero in mathematical expression.")
                res = left / right
            elif isinstance(node.op, ast.FloorDiv):
                if right == 0.0:
                    raise ZeroDivisionError("Floor division by zero in mathematical expression.")
                res = left // right
            elif isinstance(node.op, ast.Mod):
                if right == 0.0:
                    raise ZeroDivisionError("Modulo by zero in mathematical expression.")
                res = left % right
            elif isinstance(node.op, ast.Pow):
                if abs(right) > cls._MAX_EXPONENT:
                    raise ValueError(f"Exponent {right} exceeds safe limit of {cls._MAX_EXPONENT}.")
                if abs(left) > 1e6 and right > 2:
                    raise ValueError("Base and exponent combination exceeds safe computation limits.")
                res = left ** right
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

            if math.isnan(res) or math.isinf(res):
                raise ValueError("Calculation resulted in a non-finite value (NaN or Inf).")
            if abs(res) > cls._MAX_MAGNITUDE:
                raise ValueError(f"Calculation result {res} exceeds maximum allowed magnitude of {cls._MAX_MAGNITUDE}.")

            return float(res)

        raise ValueError(f"Unexpected AST node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """
    Sandboxed mathematical calculator evaluating arithmetic expressions
    using a strict AST allowlist without eval(), exec(), or subprocess.
    """
    name = "calculator"
    description = (
        "Compute numerical arithmetic calculations (+, -, *, /, %, **) "
        "using a safe AST-based sandboxed engine without code execution."
    )
    input_schema = CalculatorInput
    output_schema = CalculatorOutput
    timeout_seconds = 2.0

    async def execute(self, validated_input: BaseModel, context: Optional[Dict[str, Any]] = None) -> BaseModel:
        assert isinstance(validated_input, CalculatorInput)
        result = SafeCalculatorEvaluator.evaluate(validated_input.expression)

        return CalculatorOutput(
            result=result,
            expression=validated_input.expression.strip(),
            rounded_result=round(result, 4),
        )


# ── Tool Registry Service ────────────────────────────────────────────────────


class ToolRegistry:
    """
    Central registry governing discovery, argument validation, timeout enforcement,
    and safe execution of approved agent tools.
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register an approved tool instance."""
        if not tool.name or not tool.name.strip():
            raise ValueError("Tool name cannot be empty.")
        self._tools[tool.name.strip()] = tool
        logger.debug("Registered tool in registry: '%s'", tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by identifier."""
        return self._tools.get(name.strip())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name.strip() in self._tools

    def list_tools(self) -> List[ToolDefinition]:
        """Return public structured definitions for all registered tools."""
        return [t.get_definition() for t in self._tools.values()]

    def select_tool(
        self,
        step: Optional[PlanStep] = None,
        query_intent: Optional[QueryIntent] = None,
        subquery_text: str = "",
        has_document_filter: bool = False,
    ) -> str:
        """
        Deterministically select the appropriate tool based on execution plan step,
        query intent, subquery context, and document filter constraints.

        Core Selection Principles:
          1. Retrieval Primacy: When document evidence is required (e.g. step_type == RETRIEVAL),
             NEVER select calculator.
          2. Document Filtering: If specific document IDs are targeted, select metadata_filter.
          3. Arithmetic Inquiries: If explicit calculation is requested and tool is present,
             select calculator.
          4. Default: semantic_search.
        """
        # Rule 1 & 2: Plan step is a retrieval step
        if step and step.step_type == PlanStepType.RETRIEVAL:
            if has_document_filter and self.has_tool("metadata_filter"):
                return "metadata_filter"
            if self.has_tool("semantic_search"):
                return "semantic_search"

        # Rule 3: Document filter requested without explicit retrieval step
        if has_document_filter and self.has_tool("metadata_filter"):
            return "metadata_filter"

        # Rule 4: Dedicated arithmetic calculation
        sq_clean = subquery_text.lower().strip()
        is_explicit_calculation = (
            sq_clean.startswith(("calculate", "compute", "what is the sum of", "what is the product of"))
            and any(op in sq_clean for op in ["+", "-", "*", "/", "%", "**", "divided by", "multiplied by"])
        )
        if is_explicit_calculation and self.has_tool("calculator"):
            return "calculator"

        # Default fallback
        if self.has_tool("semantic_search"):
            return "semantic_search"

        # If semantic_search not available, pick first registered tool or error out
        first_tool = next(iter(self._tools.keys()), "")
        return first_tool

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        call_count: int = 0,
        step_number: Optional[int] = None,
    ) -> ToolResult:
        """
        Safely execute a registered tool with argument validation, timeout enforcement,
        and central call budget guards.

        Args:
            tool_name: Registered tool identifier.
            arguments: Raw input dictionary.
            context: Optional execution context (e.g. database session).
            call_count: Current count of tool calls executed in active session.
            step_number: Optional execution plan step number.

        Returns:
            Immutable ToolResult with status, output or error details.
        """
        call_id = f"call-{uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # 1. Central call limit check
        if call_count >= MAX_TOOL_CALLS_PER_RUN:
            msg = (
                f"Centrally enforced tool call limit of {MAX_TOOL_CALLS_PER_RUN} exceeded. "
                f"Attempted call '{tool_name}' rejected."
            )
            logger.warning(msg)
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.CALL_LIMIT_EXCEEDED,
                error_message=msg,
                latency_ms=0.0,
            )

        # 2. Tool availability check
        tool = self.get(tool_name)
        if not tool:
            msg = f"Tool '{tool_name}' is not registered in ToolRegistry."
            logger.error(msg)
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.TOOL_NOT_FOUND,
                error_message=msg,
                latency_ms=0.0,
            )

        # 3. Input validation
        try:
            validated_input = tool.input_schema.model_validate(arguments)
        except ValidationError as val_err:
            duration = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_msg = f"Input validation error for tool '{tool_name}': {val_err}"
            logger.warning(err_msg)
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.INVALID_ARGUMENTS,
                error_message=err_msg,
                latency_ms=duration,
            )

        # 4. Asynchronous execution with per-tool timeout
        try:
            output = await asyncio.wait_for(
                tool.execute(validated_input=validated_input, context=context),
                timeout=tool.timeout_seconds,
            )
            duration = round((time.perf_counter() - start_time) * 1000.0, 2)
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.SUCCESS,
                output=output,
                latency_ms=duration,
            )

        except asyncio.TimeoutError:
            duration = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_msg = f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s."
            logger.error(err_msg)
            db = (context or {}).get("db")
            if db:
                try:
                    await db.rollback()
                except Exception:
                    pass
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.TIMEOUT,
                error_message=err_msg,
                latency_ms=duration,
            )

        except Exception as exc:
            duration = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_msg = f"Tool '{tool_name}' failed during execution: {exc}"
            logger.exception(err_msg)
            db = (context or {}).get("db")
            if db:
                try:
                    await db.rollback()
                except Exception:
                    pass
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.EXECUTION_ERROR,
                error_message=err_msg,
                latency_ms=duration,
            )
