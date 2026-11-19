"""
backend/schemas/synthesis.py
───────────────────────────────────────────────────────────────────────────────
Pydantic v2 schemas for Phase 15: Answer Synthesis, Verification and Citations.

Provides typed models for:
  - Sentence-level grounding evaluation against retrieved source chunks
  - Semantic entailment and factual verification metrics
  - Hallucination detection flags and explanatory audit diagnostics
───────────────────────────────────────────────────────────────────────────────
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SentenceGrounding(BaseModel):
    """
    Grounding assessment for an individual sentence in the synthesized answer.
    """
    model_config = ConfigDict(frozen=True)

    sentence: str = Field(description="Sentence text evaluated for grounding.")
    cited_sources: List[str] = Field(
        default_factory=list,
        description="Source IDs (e.g. [S1]) attached to this sentence.",
    )
    is_grounded: bool = Field(
        description="Whether this sentence is factually supported by its cited sources.",
    )
    overlap_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Lexical and keyword overlap score between sentence and source text.",
    )
    unsupported_reason: Optional[str] = Field(
        default=None,
        description="Diagnostic explanation if the sentence lacks grounding.",
    )


class SynthesisVerification(BaseModel):
    """
    Comprehensive verification report produced by AnswerVerifier.
    Establishes semantic entailment and grounding integrity beyond reference presence.
    """
    model_config = ConfigDict(frozen=True)

    is_verified: bool = Field(
        description="True if the answer achieves adequate semantic grounding across its claims.",
    )
    grounding_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall proportion of grounded content (0.0 to 1.0).",
    )
    total_sentences: int = Field(default=0, ge=0)
    grounded_sentences: int = Field(default=0, ge=0)
    valid_citations: List[str] = Field(default_factory=list)
    invalid_citations: List[str] = Field(default_factory=list)
    uncited_sources: List[str] = Field(default_factory=list)
    sentence_evaluations: List[SentenceGrounding] = Field(default_factory=list)
    hallucination_warning: bool = Field(
        default=False,
        description="True if ungrounded claims or invalid citations were detected.",
    )
    grounding_explanation: str = Field(
        description="Auditable summary of the verification outcome.",
    )
