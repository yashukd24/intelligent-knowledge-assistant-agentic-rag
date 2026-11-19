"""
backend/schemas/evidence.py
───────────────────────────────────────────────────────────────────────────────
Pydantic v2 schemas for Phase 13: Evidence Evaluation and Sufficiency Decisions.

Provides structured, typed models for:
  - Chunk-level relevance and keyword overlap analysis
  - Decomposed subquery coverage evaluation
  - Multi-criteria evidence scoring and sufficiency determination
  - Standardized insufficiency reasons and explanatory diagnostics
───────────────────────────────────────────────────────────────────────────────
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SufficiencyReason(str, Enum):
    """Enumeration of deterministic evidence evaluation outcomes."""
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"
    LOW_SIMILARITY_SCORE = "LOW_SIMILARITY_SCORE"
    INSUFFICIENT_SUBQUERY_COVERAGE = "INSUFFICIENT_SUBQUERY_COVERAGE"
    CONTEXT_TOO_BRIEF = "CONTEXT_TOO_BRIEF"
    NO_RELEVANT_CONTENT = "NO_RELEVANT_CONTENT"


class ChunkRelevance(BaseModel):
    """Relevance assessment for a single retrieved chunk."""
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique identifier of the evaluated chunk.")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Vector cosine similarity score.")
    keyword_overlap_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Jaccard or token overlap between query terms and chunk text.",
    )
    composite_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted composite score combining similarity and keyword overlap.",
    )
    passes_threshold: bool = Field(
        description="Whether this chunk satisfies the minimal relevance bar.",
    )


class SubqueryCoverage(BaseModel):
    """Coverage assessment for an individual decomposed subquery."""
    model_config = ConfigDict(frozen=True)

    subquery_id: str = Field(description="Identifier or index of the subquery.")
    subquery_text: str = Field(description="Target subquery text.")
    matched_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs that provide relevant evidence for this subquery.",
    )
    highest_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Highest similarity score among matching chunks.",
    )
    is_covered: bool = Field(
        description="True if at least one chunk adequately addresses this subquery.",
    )


class EvidenceEvaluation(BaseModel):
    """
    Comprehensive evidence sufficiency report produced by EvidenceEvaluator.
    Evaluates retrieved chunks against user requirements before answer synthesis.
    """
    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregated evidence confidence score (0.0 = completely insufficient, 1.0 = fully grounded).",
    )
    is_sufficient: bool = Field(
        description="Deterministic binary decision: True if retrieval is sufficient to synthesize answer.",
    )
    reason: SufficiencyReason = Field(
        description="Categorical reason code for the sufficiency decision.",
    )
    explanation: str = Field(
        description="Human-readable and auditable explanation of the evaluation rationale.",
    )
    chunk_evaluations: List[ChunkRelevance] = Field(
        default_factory=list,
        description="Per-chunk relevance breakdowns.",
    )
    subquery_coverages: List[SubqueryCoverage] = Field(
        default_factory=list,
        description="Subquery coverage analysis for compound/multi-part requests.",
    )
    coverage_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Proportion of decomposed subqueries covered by retrieved evidence.",
    )
    min_similarity_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score required to qualify evidence.",
    )
    total_chunks_evaluated: int = Field(
        default=0,
        ge=0,
        description="Total number of chunks evaluated.",
    )
    qualifying_chunk_count: int = Field(
        default=0,
        ge=0,
        description="Number of chunks that met or exceeded the qualification threshold.",
    )
