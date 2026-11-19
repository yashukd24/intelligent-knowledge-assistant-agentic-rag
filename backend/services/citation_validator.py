"""
backend/services/citation_validator.py
─────────────────────────────────────────────────────────────
CitationValidator: Parses generated LLM answers for source citations
([S1], [S2], etc.), validates them against the authoritative retrieved
source set, identifies invalid or fabricated references, and links valid
citations back to their underlying source document metadata.

Important Semantic Limitation:
  Citation validation strictly establishes *reference integrity* (verifying
  that a cited identifier exists in the retrieved context). It does NOT
  guarantee *semantic entailment* (proving that the cited document text
  actually supports or proves the accompanying statement). Semantic
  entailment and grounding verification belong to Phase 15.
─────────────────────────────────────────────────────────────
"""

import logging
import re
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.rag import SourceReference

logger = logging.getLogger(__name__)

# Regular expression to match standard source identifiers: [S1], [S2], [S12], etc.
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class CitationValidationResult(BaseModel):
    """
    Structured outcome of parsing and validating citations in an LLM answer.
    """
    cited_sources: List[str] = Field(
        default_factory=list,
        description="All citation identifiers found in the answer in order of first appearance.",
    )
    valid_citations: List[str] = Field(
        default_factory=list,
        description="Citation identifiers that map to an actual retrieved source in context.",
    )
    invalid_citations: List[str] = Field(
        default_factory=list,
        description="Fabricated or unknown citation identifiers not present in retrieved context.",
    )
    uncited_sources: List[str] = Field(
        default_factory=list,
        description="Retrieved sources that were provided in context but not cited in the answer.",
    )
    valid_source_references: List[SourceReference] = Field(
        default_factory=list,
        description="Underlying source references corresponding to valid citations.",
    )
    has_citations: bool = Field(
        default=False,
        description="True if at least one citation was detected in the answer.",
    )
    has_invalid_citations: bool = Field(
        default=False,
        description="True if one or more cited identifiers do not exist in the source set.",
    )

    model_config = ConfigDict(from_attributes=True)


class CitationValidator:
    """
    Validates citation integrity in LLM-generated answers.
    """

    def validate_citations(
        self,
        answer_text: str,
        sources: Optional[List[SourceReference]] = None,
    ) -> CitationValidationResult:
        """
        Extract and validate citations from an answer string.

        Args:
            answer_text: The LLM-generated answer text.
            sources: List of valid SourceReference objects provided in context.

        Returns:
            CitationValidationResult with categorized citations and metadata.
        """
        if not answer_text or not answer_text.strip():
            return CitationValidationResult()

        sources_list = sources or []
        source_map: Dict[str, SourceReference] = {s.source_id: s for s in sources_list}
        known_source_ids: Set[str] = set(source_map.keys())

        # Extract all citation tags preserving first occurrence order
        raw_matches = CITATION_PATTERN.findall(answer_text)
        cited_sources: List[str] = []
        seen: Set[str] = set()

        for num_str in raw_matches:
            tag = f"[S{num_str}]"
            if tag not in seen:
                seen.add(tag)
                cited_sources.append(tag)

        # Categorize citations into valid vs invalid
        valid_citations: List[str] = []
        invalid_citations: List[str] = []
        valid_source_references: List[SourceReference] = []

        for tag in cited_sources:
            if tag in known_source_ids:
                valid_citations.append(tag)
                valid_source_references.append(source_map[tag])
            else:
                invalid_citations.append(tag)
                logger.warning("Invalid or hallucinated citation detected in LLM answer: %s", tag)

        # Identify uncited sources provided in context
        uncited_sources = [s.source_id for s in sources_list if s.source_id not in valid_citations]

        return CitationValidationResult(
            cited_sources=cited_sources,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            uncited_sources=uncited_sources,
            valid_source_references=valid_source_references,
            has_citations=len(cited_sources) > 0,
            has_invalid_citations=len(invalid_citations) > 0,
        )
