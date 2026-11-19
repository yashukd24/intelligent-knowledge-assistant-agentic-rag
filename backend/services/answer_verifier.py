"""
backend/services/answer_verifier.py
───────────────────────────────────────────────────────────────────────────────
Phase 15: Answer Synthesis, Verification and Citations.

Implements semantic grounding verification and citation generation:
  - Validates citation reference integrity
  - Evaluates sentence-level factual alignment with source chunks
  - Detects ungrounded numeric or factual claims (hallucination defense)
  - Emits auditable SynthesisVerification metrics
  - Plain Python execution without external LLM dependencies
───────────────────────────────────────────────────────────────────────────────
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from backend.schemas.rag import SourceReference
from backend.schemas.synthesis import SentenceGrounding, SynthesisVerification
from backend.services.citation_validator import CITATION_PATTERN, CitationValidator

logger = logging.getLogger(__name__)

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

_CONVERSATIONAL_PREFIXES = [
    "based on the provided documents",
    "according to the provided text",
    "as mentioned in the context",
    "the retrieved documents state",
]


def _extract_keywords(text: str) -> Set[str]:
    """Extract lowercased alphanumeric keywords excluding common stop words."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _extract_numerics(text: str) -> Set[str]:
    """Extract numbers, dollar values, percentages, and year representations."""
    matches = re.findall(r"\b(?:\$?\d+(?:\.\d+)?[MBk%]?(?:st|nd|rd|th)?)\b", text)
    return set(matches)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while keeping citation tags attached."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned = [s.strip() for s in raw_sentences if s.strip()]
    return cleaned if cleaned else ([text.strip()] if text.strip() else [])


class AnswerVerifier:
    """
    Verifies semantic grounding, factual claim alignment, and citation validity
    for synthesized answers.
    """

    def __init__(
        self,
        citation_validator: Optional[CitationValidator] = None,
        min_grounding_threshold: float = 0.65,
        min_overlap_ratio: float = 0.15,
    ):
        self.citation_validator = citation_validator or CitationValidator()
        self.min_grounding_threshold = min_grounding_threshold
        self.min_overlap_ratio = min_overlap_ratio

    def verify(
        self,
        answer_text: str,
        sources: Optional[List[SourceReference]] = None,
        chunks: Optional[List[Any]] = None,
        assembled_context: str = "",
    ) -> SynthesisVerification:
        """
        Perform multi-level answer verification:
          1. Citation reference integrity check.
          2. Sentence-level semantic grounding & token overlap.
          3. Numeric & entity consistency check (factual verification).
          4. Global grounding score computation and hallucination warnings.

        Args:
            answer_text: Synthesized answer string.
            sources: List of SourceReference objects provided in context.
            assembled_context: Raw assembled prompt context.

        Returns:
            SynthesisVerification report.
        """
        if not answer_text or not answer_text.strip():
            return SynthesisVerification(
                is_verified=False,
                grounding_score=0.0,
                total_sentences=0,
                grounded_sentences=0,
                hallucination_warning=False,
                grounding_explanation="Answer text is empty.",
            )

        sources_list = sources or []
        source_map: Dict[str, SourceReference] = {s.source_id: s for s in sources_list}

        # 1. Base Citation Validation
        citation_result = self.citation_validator.validate_citations(
            answer_text=answer_text,
            sources=sources_list,
        )

        # Build mapping from source_id -> chunk text
        source_text_map: Dict[str, str] = {}
        if chunks:
            chunk_by_id = {getattr(c, "chunk_id", ""): getattr(c, "text", "") for c in chunks}
            for s in sources_list:
                if s.chunk_id in chunk_by_id:
                    source_text_map[s.source_id] = chunk_by_id[s.chunk_id]

        sentences = _split_into_sentences(answer_text)
        sentence_evals: List[SentenceGrounding] = []
        grounded_count = 0

        for sent in sentences:
            sent_citations = [f"[S{num}]" for num in CITATION_PATTERN.findall(sent)]
            sent_keywords = _extract_keywords(sent)
            sent_numerics = _extract_numerics(sent)

            # Check conversational sentence exemption
            is_conversational = any(
                sent.lower().startswith(p) for p in _CONVERSATIONAL_PREFIXES
            ) and len(sent_keywords) <= 4

            if is_conversational and not sent_citations:
                sentence_evals.append(
                    SentenceGrounding(
                        sentence=sent,
                        cited_sources=[],
                        is_grounded=True,
                        overlap_score=1.0,
                        unsupported_reason=None,
                    )
                )
                grounded_count += 1
                continue

            # Case: Sentence contains citations
            if sent_citations:
                invalid_in_sent = [c for c in sent_citations if c in citation_result.invalid_citations]
                if invalid_in_sent:
                    sentence_evals.append(
                        SentenceGrounding(
                            sentence=sent,
                            cited_sources=sent_citations,
                            is_grounded=False,
                            overlap_score=0.0,
                            unsupported_reason=f"Contains invalid citation(s): {invalid_in_sent}",
                        )
                    )
                    continue

                # Aggregate text of valid cited sources
                cited_texts = []
                for c in sent_citations:
                    if c in source_text_map and source_text_map[c]:
                        cited_texts.append(source_text_map[c])
                    elif c in source_map and hasattr(source_map[c], "text_preview") and getattr(source_map[c], "text_preview", None):
                        cited_texts.append(source_map[c].text_preview)

                cited_text = " ".join(cited_texts)
                # Fallback to assembled context if preview was empty
                if not cited_text.strip():
                    cited_text = assembled_context

                source_keywords = _extract_keywords(cited_text)
                source_numerics = _extract_numerics(cited_text)

                overlap = (
                    len(sent_keywords.intersection(source_keywords)) / len(sent_keywords)
                    if sent_keywords
                    else 0.0
                )
                overlap = round(min(1.0, max(0.0, overlap)), 4)

                # Check if all numbers in sentence exist in the source text
                missing_numerics = sent_numerics - source_numerics
                has_numeric_mismatch = bool(missing_numerics)

                if has_numeric_mismatch:
                    is_grounded = False
                    reason = f"Contains ungrounded numeric claim(s): {list(missing_numerics)}"
                elif overlap >= self.min_overlap_ratio:
                    is_grounded = True
                    reason = None
                else:
                    is_grounded = False
                    reason = f"Low semantic overlap with cited source ({overlap:.1%} < {self.min_overlap_ratio:.1%})"

                sentence_evals.append(
                    SentenceGrounding(
                        sentence=sent,
                        cited_sources=sent_citations,
                        is_grounded=is_grounded,
                        overlap_score=overlap,
                        unsupported_reason=reason,
                    )
                )
                if is_grounded:
                    grounded_count += 1

            # Case: Substantive sentence without citations
            else:
                if len(sent_keywords) >= 4:
                    sentence_evals.append(
                        SentenceGrounding(
                            sentence=sent,
                            cited_sources=[],
                            is_grounded=False,
                            overlap_score=0.0,
                            unsupported_reason="Factual sentence lacks source citation.",
                        )
                    )
                else:
                    # Very short transition sentence
                    sentence_evals.append(
                        SentenceGrounding(
                            sentence=sent,
                            cited_sources=[],
                            is_grounded=True,
                            overlap_score=0.5,
                            unsupported_reason=None,
                        )
                    )
                    grounded_count += 1

        total_sents = len(sentences)
        score = round(grounded_count / total_sents, 4) if total_sents > 0 else 0.0
        has_invalid = bool(citation_result.invalid_citations)
        has_warning = has_invalid or (score < self.min_grounding_threshold)
        is_verified = not has_invalid and (score >= self.min_grounding_threshold)

        explanation = (
            f"Verification passed: {grounded_count}/{total_sents} sentences grounded ({score:.0%}). "
            f"All {len(citation_result.valid_citations)} citations verified."
            if is_verified
            else f"Verification warning: grounding score {score:.0%} below threshold {self.min_grounding_threshold:.0%} "
                 f"or invalid citations detected ({citation_result.invalid_citations})."
        )

        logger.info("Answer verification completed: is_verified=%s, score=%.2f", is_verified, score)

        return SynthesisVerification(
            is_verified=is_verified,
            grounding_score=score,
            total_sentences=total_sents,
            grounded_sentences=grounded_count,
            valid_citations=citation_result.valid_citations,
            invalid_citations=citation_result.invalid_citations,
            uncited_sources=citation_result.uncited_sources,
            sentence_evaluations=sentence_evals,
            hallucination_warning=has_warning,
            grounding_explanation=explanation,
        )
