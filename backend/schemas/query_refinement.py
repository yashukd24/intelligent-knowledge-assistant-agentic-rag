"""
backend/schemas/query_refinement.py
───────────────────────────────────────────────────────────────────────────────
Pydantic v2 schemas for Phase 14: Query Refinement and Adaptive Re-retrieval.

Provides typed models for:
  - Adaptive refinement strategies (expansion, relaxation, decomposition targeting)
  - Parameter adjustments (top_k adjustments, threshold relaxation)
  - Audit trail of refinement attempts per agent session
───────────────────────────────────────────────────────────────────────────────
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RefinementStrategy(str, Enum):
    """Categorical strategy for adaptive query reformulation."""
    TARGET_UNCOVERED_SUBQUERIES = "TARGET_UNCOVERED_SUBQUERIES"
    RELAX_SIMILARITY_THRESHOLD = "RELAX_SIMILARITY_THRESHOLD"
    EXPAND_SEARCH_TERMS = "EXPAND_SEARCH_TERMS"
    BROADEN_RETRIEVAL_SCOPE = "BROADEN_RETRIEVAL_SCOPE"


class RefinedQuery(BaseModel):
    """
    Structured record of a single adaptive refinement operation.
    """
    model_config = ConfigDict(frozen=True)

    iteration: int = Field(ge=1, description="Sequential refinement attempt index.")
    strategy: RefinementStrategy = Field(description="Selected refinement strategy.")
    original_query: str = Field(description="Query prior to refinement.")
    refined_query_text: str = Field(description="Reformulated search text.")
    targeted_subquery_ids: List[str] = Field(
        default_factory=list,
        description="IDs of subqueries specifically targeted by this refinement.",
    )
    adjusted_top_k: Optional[int] = Field(
        default=None,
        description="Adjusted top_k parameter if altered for re-retrieval.",
    )
    adjusted_similarity_threshold: Optional[float] = Field(
        default=None,
        description="Adjusted similarity threshold if relaxed for re-retrieval.",
    )
    rationale: str = Field(description="Auditable explanation of why this refinement was chosen.")
