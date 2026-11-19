"""
backend/services/query_refiner.py
───────────────────────────────────────────────────────────────────────────────
Phase 14: Query Refinement and Adaptive Re-retrieval.

Implements an adaptive query refiner that inspects evidence evaluation reports
and formulates targeted re-retrieval requests when retrieved evidence is
insufficient.

Core Capabilities:
  - Targeted subquery isolation for multi-part queries with partial evidence
  - Dynamic similarity threshold relaxation and top_k expansion
  - Term broadening for queries that yielded empty or near-empty results
  - Strict plain-Python execution without external dependencies
───────────────────────────────────────────────────────────────────────────────
"""

import logging
import re
from typing import List, Optional

from backend.schemas.evidence import EvidenceEvaluation, SufficiencyReason
from backend.schemas.query_plan import SubQuery
from backend.schemas.query_refinement import RefinedQuery, RefinementStrategy

logger = logging.getLogger(__name__)


class QueryRefiner:
    """
    Adaptive query refinement engine that reformulates search parameters
    and queries when initial retrieval is insufficient.
    """

    def __init__(self, max_refinements: int = 2):
        self.max_refinements = max_refinements

    def can_refine(
        self,
        current_iteration: int,
        evaluation: EvidenceEvaluation,
        max_iterations: Optional[int] = None,
    ) -> bool:
        """
        Determine whether another refinement iteration is permissible and beneficial.
        """
        if max_iterations is not None:
            limit = min(max_iterations, self.max_refinements)
        else:
            limit = self.max_refinements
        if current_iteration >= limit:
            logger.info("Refinement limit reached: iteration %d >= limit %d.", current_iteration, limit)
            return False

        if evaluation.is_sufficient:
            return False

        # All defined insufficiency reasons are eligible for adaptive refinement
        return True

    def refine(
        self,
        current_iteration: int,
        original_query: str,
        evaluation: EvidenceEvaluation,
        subqueries: Optional[List[SubQuery]] = None,
        current_top_k: int = 5,
        current_similarity_threshold: Optional[float] = None,
    ) -> RefinedQuery:
        """
        Synthesize an adaptive refinement plan based on the evaluation diagnostics.

        Args:
            current_iteration: 0-indexed count of prior refinements completed.
            original_query: The user's original query.
            evaluation: Phase 13 EvidenceEvaluation diagnostics report.
            subqueries: List of decomposed subqueries (if available).
            current_top_k: Existing retrieval candidate count.
            current_similarity_threshold: Existing similarity threshold.

        Returns:
            RefinedQuery with new text, strategy, and adjusted parameters.
        """
        iteration_num = current_iteration + 1
        base_threshold = current_similarity_threshold if current_similarity_threshold is not None else 0.50

        # ── 1. Strategy: Target Uncovered Subqueries ─────────────────────────
        if evaluation.reason == SufficiencyReason.INSUFFICIENT_SUBQUERY_COVERAGE and evaluation.subquery_coverages:
            uncovered = [sc for sc in evaluation.subquery_coverages if not sc.is_covered]
            if uncovered:
                targeted_ids = [sc.subquery_id for sc in uncovered]
                targeted_texts = [sc.subquery_text for sc in uncovered]
                refined_text = " ".join(targeted_texts)
                logger.info(
                    "Refinement (iteration %d): Targeting %d uncovered subquery(ies): %s",
                    iteration_num,
                    len(uncovered),
                    targeted_ids,
                )
                return RefinedQuery(
                    iteration=iteration_num,
                    strategy=RefinementStrategy.TARGET_UNCOVERED_SUBQUERIES,
                    original_query=original_query,
                    refined_query_text=refined_text,
                    targeted_subquery_ids=targeted_ids,
                    adjusted_top_k=min(current_top_k + 2, 15),
                    adjusted_similarity_threshold=round(max(0.25, base_threshold - 0.10), 4),
                    rationale=(
                        f"Initial retrieval missed evidence for subqueries: {targeted_ids}. "
                        f"Refining search to specifically target missing requirements."
                    ),
                )

        # ── 2. Strategy: Relax Similarity Threshold & Expand Top-K ───────────
        if evaluation.reason in (SufficiencyReason.LOW_SIMILARITY_SCORE, SufficiencyReason.NO_RELEVANT_CONTENT):
            adjusted_sim = round(max(0.20, base_threshold - 0.15), 4)
            adjusted_k = min(current_top_k + 4, 15)
            # Remove minor punctuation that might degrade term match
            cleaned = re.sub(r"[?!.,;:\'\"]+", " ", original_query).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)

            logger.info(
                "Refinement (iteration %d): Relaxing threshold from %.2f to %.2f, top_k from %d to %d.",
                iteration_num,
                base_threshold,
                adjusted_sim,
                current_top_k,
                adjusted_k,
            )
            return RefinedQuery(
                iteration=iteration_num,
                strategy=RefinementStrategy.RELAX_SIMILARITY_THRESHOLD,
                original_query=original_query,
                refined_query_text=cleaned,
                targeted_subquery_ids=[],
                adjusted_top_k=adjusted_k,
                adjusted_similarity_threshold=adjusted_sim,
                rationale=(
                    f"Retrieved chunks had low scores (< {base_threshold:.2f}). "
                    f"Relaxed threshold to {adjusted_sim:.2f} and expanded candidate window to {adjusted_k}."
                ),
            )

        # ── 3. Strategy: Broaden Scope for Empty / Brief Context ─────────────
        # Strip conversational prefix if present, extract salient core
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", original_query)
        broad_text = " ".join(tokens) if tokens else original_query
        adjusted_sim = round(max(0.20, base_threshold - 0.15), 4)
        adjusted_k = min(current_top_k + 5, 15)

        logger.info(
            "Refinement (iteration %d): Broadening search scope to: '%s'",
            iteration_num,
            broad_text,
        )
        return RefinedQuery(
            iteration=iteration_num,
            strategy=RefinementStrategy.BROADEN_RETRIEVAL_SCOPE,
            original_query=original_query,
            refined_query_text=broad_text,
            targeted_subquery_ids=[],
            adjusted_top_k=adjusted_k,
            adjusted_similarity_threshold=adjusted_sim,
            rationale=(
                f"Retrieved evidence was empty or insufficient ({evaluation.reason.value}). "
                f"Broadening search terms and relaxing constraints."
            ),
        )
