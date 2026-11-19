"""
backend/services/evidence_evaluator.py
───────────────────────────────────────────────────────────────────────────────
Phase 13: Evidence Evaluation and Sufficiency Decisions.

Implements a deterministic, rule-based evidence evaluator that assesses whether
retrieved document chunks are sufficient to synthesize a faithful, grounded
answer.

Key Capabilities:
  - Multi-criteria chunk evaluation (vector similarity + keyword density overlap)
  - Subquery coverage auditing for decomposed complex queries
  - Deterministic sufficiency decision gating with categorical reason codes
  - Strict plain-Python execution without external LLM calls or LangChain
───────────────────────────────────────────────────────────────────────────────
"""

import logging
import re
from typing import List, Optional, Set

from backend.schemas.evidence import (
    ChunkRelevance,
    EvidenceEvaluation,
    SubqueryCoverage,
    SufficiencyReason,
)
from backend.schemas.query_plan import SubQuery
from backend.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Lightweight stopword list for plain-Python keyword overlap computation
_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "were", "weren't", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
}


def _extract_keywords(text: str) -> Set[str]:
    """Extract lowercased alphanumeric keywords excluding common stop words."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


class EvidenceEvaluator:
    """
    Evaluates retrieved chunks against the original query and execution plan
    to deterministically decide if evidence is sufficient for answer synthesis.
    """

    def __init__(
        self,
        default_similarity_threshold: float = 0.50,
        min_coverage_ratio: float = 0.60,
        min_context_chars: int = 30,
        weight_similarity: float = 0.75,
        weight_keyword: float = 0.25,
    ):
        self.default_similarity_threshold = default_similarity_threshold
        self.min_coverage_ratio = min_coverage_ratio
        self.min_context_chars = min_context_chars
        self.weight_similarity = weight_similarity
        self.weight_keyword = weight_keyword

    def evaluate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        subqueries: Optional[List[SubQuery]] = None,
        assembled_context: str = "",
        similarity_threshold: Optional[float] = None,
    ) -> EvidenceEvaluation:
        """
        Evaluate retrieved evidence across multiple dimensions:
          1. Chunk-level relevance scores.
          2. Decomposed subquery coverage.
          3. Overall sufficiency gating.

        Args:
            query: The user's original or normalized query.
            chunks: Deduplicated list of RetrievedChunk objects.
            subqueries: Optional list of decomposed subqueries from Phase 11.
            assembled_context: Combined text context formatted for prompt.
            similarity_threshold: Optional override for minimum similarity bar.

        Returns:
            EvidenceEvaluation report with is_sufficient decision and diagnostics.
        """
        sim_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )

        query_keywords = _extract_keywords(query)

        # ── 1. Empty or Missing Retrieval Gate ───────────────────────────────
        if not chunks:
            logger.info("Evidence evaluation: No chunks retrieved (EMPTY_RETRIEVAL).")
            return EvidenceEvaluation(
                overall_score=0.0,
                is_sufficient=False,
                reason=SufficiencyReason.EMPTY_RETRIEVAL,
                explanation="No document chunks were retrieved for the query.",
                chunk_evaluations=[],
                subquery_coverages=[],
                coverage_ratio=0.0,
                min_similarity_threshold=sim_threshold,
                total_chunks_evaluated=0,
                qualifying_chunk_count=0,
            )

        if not assembled_context.strip() or len(assembled_context.strip()) < self.min_context_chars:
            logger.info("Evidence evaluation: Context is too brief (< %d chars).", self.min_context_chars)
            return EvidenceEvaluation(
                overall_score=0.0,
                is_sufficient=False,
                reason=SufficiencyReason.CONTEXT_TOO_BRIEF,
                explanation=(
                    f"Assembled context contains only {len(assembled_context.strip())} characters, "
                    f"which is below the minimum threshold of {self.min_context_chars} characters."
                ),
                chunk_evaluations=[],
                subquery_coverages=[],
                coverage_ratio=0.0,
                min_similarity_threshold=sim_threshold,
                total_chunks_evaluated=len(chunks),
                qualifying_chunk_count=0,
            )

        # ── 2. Per-Chunk Relevance Scoring ───────────────────────────────────
        chunk_evals: List[ChunkRelevance] = []
        qualifying_chunks: List[RetrievedChunk] = []

        for c in chunks:
            chunk_keywords = _extract_keywords(c.text)
            if query_keywords and chunk_keywords:
                overlap = len(query_keywords.intersection(chunk_keywords)) / len(query_keywords)
            else:
                overlap = 0.0
            overlap = round(min(1.0, max(0.0, overlap)), 4)

            # Composite weighted score
            sim = round(float(c.similarity_score), 4)
            comp = round(
                (self.weight_similarity * sim) + (self.weight_keyword * overlap),
                4,
            )
            # Qualifies if similarity meets threshold OR composite score is high.
            # Anti-false-positive check: If the query has multiple distinct keywords (>=2)
            # but this chunk has zero lexical overlap (0.0), require high semantic similarity
            # (>= 0.70) to prevent borderline boilerplate/noise from being declared qualifying.
            if len(query_keywords) >= 2 and overlap == 0.0:
                passes = sim >= max(0.70, sim_threshold + 0.15)
            else:
                passes = sim >= sim_threshold or comp >= sim_threshold

            chunk_eval = ChunkRelevance(
                chunk_id=c.chunk_id,
                similarity_score=sim,
                keyword_overlap_ratio=overlap,
                composite_score=comp,
                passes_threshold=passes,
            )
            chunk_evals.append(chunk_eval)
            if passes:
                qualifying_chunks.append(c)

        max_sim = max([ce.similarity_score for ce in chunk_evals], default=0.0)
        max_comp = max([ce.composite_score for ce in chunk_evals], default=0.0)

        # ── 3. Low Similarity Gate ───────────────────────────────────────────
        if max_sim < sim_threshold and max_comp < sim_threshold:
            logger.info(
                "Evidence evaluation: Max score (sim=%.3f, comp=%.3f) below threshold %.3f.",
                max_sim,
                max_comp,
                sim_threshold,
            )
            return EvidenceEvaluation(
                overall_score=round(max_sim, 4),
                is_sufficient=False,
                reason=SufficiencyReason.LOW_SIMILARITY_SCORE,
                explanation=(
                    f"Highest retrieved similarity score ({max_sim:.3f}) is below "
                    f"the required threshold of {sim_threshold:.3f}."
                ),
                chunk_evaluations=chunk_evals,
                subquery_coverages=[],
                coverage_ratio=0.0,
                min_similarity_threshold=sim_threshold,
                total_chunks_evaluated=len(chunks),
                qualifying_chunk_count=len(qualifying_chunks),
            )

        # ── 4. Decomposed Subquery Coverage Evaluation ───────────────────────
        subquery_coverages: List[SubqueryCoverage] = []
        coverage_ratio = 1.0

        if subqueries and len(subqueries) > 1:
            covered_count = 0
            for sq in subqueries:
                sq_keywords = _extract_keywords(sq.text)
                matching_chunk_ids: List[str] = []
                sq_max_sim = 0.0

                for c in chunks:
                    c_sim = float(c.similarity_score)
                    c_keywords = _extract_keywords(c.text)
                    sq_overlap = (
                        len(sq_keywords.intersection(c_keywords)) / len(sq_keywords)
                        if sq_keywords and c_keywords
                        else 0.0
                    )

                    # A chunk covers a specific subquery if it shares key terminology
                    # with the subquery and has adequate similarity
                    has_keyword_match = (
                        bool(sq_keywords.intersection(c_keywords)) if sq_keywords else True
                    )
                    if has_keyword_match and (c_sim >= sim_threshold * 0.70):
                        matching_chunk_ids.append(c.chunk_id)
                        if c_sim > sq_max_sim:
                            sq_max_sim = c_sim

                is_covered = len(matching_chunk_ids) > 0 and sq_max_sim >= (sim_threshold * 0.80)
                if is_covered:
                    covered_count += 1

                subquery_coverages.append(
                    SubqueryCoverage(
                        subquery_id=sq.id,
                        subquery_text=sq.text,
                        matched_chunk_ids=matching_chunk_ids,
                        highest_similarity=round(sq_max_sim, 4),
                        is_covered=is_covered,
                    )
                )

            coverage_ratio = round(covered_count / len(subqueries), 4)

            # Subquery coverage gating
            if coverage_ratio < self.min_coverage_ratio:
                logger.info(
                    "Evidence evaluation: Subquery coverage %.2f < %.2f (INSUFFICIENT_SUBQUERY_COVERAGE).",
                    coverage_ratio,
                    self.min_coverage_ratio,
                )
                return EvidenceEvaluation(
                    overall_score=round(max_sim * coverage_ratio, 4),
                    is_sufficient=False,
                    reason=SufficiencyReason.INSUFFICIENT_SUBQUERY_COVERAGE,
                    explanation=(
                        f"Retrieved evidence covers only {covered_count}/{len(subqueries)} "
                        f"subqueries ({coverage_ratio:.0%}), which is below the minimum required "
                        f"coverage of {self.min_coverage_ratio:.0%}."
                    ),
                    chunk_evaluations=chunk_evals,
                    subquery_coverages=subquery_coverages,
                    coverage_ratio=coverage_ratio,
                    min_similarity_threshold=sim_threshold,
                    total_chunks_evaluated=len(chunks),
                    qualifying_chunk_count=len(qualifying_chunks),
                )

        # ── 5. Qualifying Chunk Count Gate ───────────────────────────────────
        if len(qualifying_chunks) == 0:
            return EvidenceEvaluation(
                overall_score=round(max_sim, 4),
                is_sufficient=False,
                reason=SufficiencyReason.NO_RELEVANT_CONTENT,
                explanation="No retrieved chunks satisfied the relevance qualification criteria.",
                chunk_evaluations=chunk_evals,
                subquery_coverages=subquery_coverages,
                coverage_ratio=coverage_ratio,
                min_similarity_threshold=sim_threshold,
                total_chunks_evaluated=len(chunks),
                qualifying_chunk_count=0,
            )

        # ── 6. Sufficient Evidence Confirmed ─────────────────────────────────
        overall_score = round(max_sim * coverage_ratio, 4)
        logger.info(
            "Evidence evaluation: SUFFICIENT_EVIDENCE (score=%.4f, %d qualifying chunks).",
            overall_score,
            len(qualifying_chunks),
        )

        return EvidenceEvaluation(
            overall_score=overall_score,
            is_sufficient=True,
            reason=SufficiencyReason.SUFFICIENT_EVIDENCE,
            explanation=(
                f"Retrieved {len(qualifying_chunks)} qualifying source chunk(s) "
                f"with maximum score {max_sim:.3f} and subquery coverage {coverage_ratio:.0%}."
            ),
            chunk_evaluations=chunk_evals,
            subquery_coverages=subquery_coverages,
            coverage_ratio=coverage_ratio,
            min_similarity_threshold=sim_threshold,
            total_chunks_evaluated=len(chunks),
            qualifying_chunk_count=len(qualifying_chunks),
        )
