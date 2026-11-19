"""
backend/evaluation/evaluation_models.py
─────────────────────────────────────────────────────────────
Pydantic v2 data models for retrieval evaluation test cases,
per-query results, aggregate quality metrics, and structured reports.
─────────────────────────────────────────────────────────────
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationTestCase(BaseModel):
    """
    A single ground-truth retrieval evaluation test case.
    """
    case_id: str = Field(..., description="Unique deterministic test case identifier.")
    query: str = Field(..., description="Natural language search query.")
    collection_id: str = Field(..., description="Target knowledge collection UUID string.")
    expected_chunk_ids: List[str] = Field(
        ...,
        description="Explicit ground-truth relevant chunk IDs supplied by dataset author.",
    )
    expected_document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of ground-truth relevant document IDs.",
    )
    query_type: str = Field(
        default="factual",
        description=(
            "Category of test case: factual, conceptual, definition, multi-part, "
            "cross-chunk, paraphrase, limited-info, unanswerable, similar-terms, doc-specific."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of what this evaluation case tests.",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("case_id cannot be empty or whitespace only.")
        return v.strip()

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query cannot be empty or whitespace only.")
        return v.strip()


class QueryEvaluationResult(BaseModel):
    """
    Evaluation metrics and candidates accounted for a single evaluation query.
    """
    case_id: str
    query: str
    query_type: str
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    expected_chunk_ids: List[str] = Field(default_factory=list)
    relevant_retrieved_chunk_ids: List[str] = Field(default_factory=list)
    precision_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    recall_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    reciprocal_rank: float = Field(default=0.0, ge=0.0, le=1.0)
    hit: bool = Field(default=False)
    latency_ms: float = Field(default=0.0, ge=0.0)
    total_candidates: int = Field(default=0, ge=0)
    returned_results: int = Field(default=0, ge=0)
    rejected_candidates: int = Field(default=0, ge=0)
    accepted_candidates: int = Field(default=0, ge=0)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AggregateMetrics(BaseModel):
    """
    Summary quality, ranking, and latency metrics across an evaluation dataset.
    """
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    hit_rate_at_k: float = Field(ge=0.0, le=1.0, description="Hit Rate @ K")
    empty_result_count: int = Field(ge=0)
    empty_result_rate: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)
    median_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    total_candidates_sum: int = Field(ge=0)
    rejected_candidates_sum: int = Field(ge=0)
    accepted_candidates_sum: int = Field(ge=0)
    total_queries: int = Field(ge=0)
    successful_queries: int = Field(ge=0)
    failed_queries: int = Field(ge=0)
    by_query_type: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Metrics breakdown grouped by query type.",
    )

    model_config = ConfigDict(from_attributes=True)


class EvaluationReport(BaseModel):
    """
    Structured Pydantic report representing a complete evaluation run.
    """
    run_id: str = Field(default_factory=lambda: f"eval_{uuid4().hex[:12]}")
    dataset_name: str
    dataset_size: int = Field(ge=0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    top_k: int = Field(ge=1, le=100)
    similarity_threshold: Optional[float] = None
    warmup_latency_ms: Optional[float] = None
    query_results: List[QueryEvaluationResult] = Field(default_factory=list)
    aggregate_metrics: AggregateMetrics
    failed_cases: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    hardware_info: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    def save_json(self, file_path: Path) -> Path:
        """Save report to a JSON file on disk."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2), encoding="utf-8")
        return target
