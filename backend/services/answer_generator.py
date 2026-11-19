"""
backend/services/answer_generator.py
─────────────────────────────────────────────────────────────
AnswerGenerator: Coordinates LLM completion generation from
a prepared RAG prompt using an injected LLMProvider, performs
citation validation on the generated text, and ensures zero LLM
calls are made when no supporting evidence exists.
─────────────────────────────────────────────────────────────
"""

from enum import Enum
import logging
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.rag import SourceReference
from backend.services.citation_validator import CitationValidationResult, CitationValidator
from backend.services.llm_provider import (
    LLMAuthenticationError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    TokenUsage,
)

logger = logging.getLogger(__name__)

STANDARD_INSUFFICIENT_EVIDENCE_ANSWER = (
    "I couldn't find this information in the uploaded college documents. Please upload the relevant document or contact the college administration."
)


class AnswerStatus(str, Enum):
    """Categorized status of the RAG answer generation attempt."""
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class AnswerGenerationResult(BaseModel):
    """
    Structured outcome of an answer generation attempt.
    """
    answer: str = Field(..., description="Generated answer text or structured refusal.")
    status: AnswerStatus = Field(..., description="Outcome status of the generation.")
    provider: str = Field(..., description="LLM provider name.")
    model: str = Field(..., description="Model identifier used.")
    usage: Optional[TokenUsage] = Field(default=None, description="Token consumption details.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Inference latency in milliseconds.")
    citations: CitationValidationResult = Field(
        default_factory=CitationValidationResult,
        description="Parsed and validated citations.",
    )
    error_message: Optional[str] = Field(default=None, description="Error message if generation failed.")

    model_config = ConfigDict(from_attributes=True)


class AnswerGenerator:
    """
    Executes grounded answer generation using an injected LLMProvider.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        citation_validator: Optional[CitationValidator] = None,
    ):
        """
        Initialize the AnswerGenerator.

        Args:
            llm_provider: Injected LLM completion provider (e.g. AnthropicProvider or mock).
            citation_validator: CitationValidator instance for parsing and verifying source references.
        """
        self.llm_provider = llm_provider
        self.citation_validator = citation_validator or CitationValidator()

    async def generate_answer(
        self,
        prompt: str,
        sources: Optional[List[SourceReference]] = None,
        has_evidence: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AnswerGenerationResult:
        """
        Generate a source-grounded answer from a prepared RAG prompt.

        Rules:
          1. If has_evidence is False or sources is empty, return an explicit
             insufficient-evidence answer without invoking the LLM provider.
          2. Otherwise, call the injected LLMProvider asynchronously.
          3. Validate and categorize citations in the generated answer text.
          4. Map provider errors (timeout, rate limit, authentication) into
             explicit AnswerStatus values without crashing or leaking credentials.

        Args:
            prompt: Assembled prompt from PromptBuilder.
            sources: List of SourceReference objects included in the context.
            has_evidence: Boolean indicating whether relevant context exists.
            max_tokens: Optional override for max output tokens.
            temperature: Optional override for generation temperature.

        Returns:
            AnswerGenerationResult model.
        """
        sources_list = sources or []

        # 1. Zero LLM call on empty evidence to eliminate unnecessary cost and hallucination risk
        if not has_evidence or not sources_list:
            logger.info("Zero retrieved evidence provided. Returning clean insufficient evidence answer.")
            return AnswerGenerationResult(
                answer=STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                provider="rule_based",
                model="none",
                usage=None,
                latency_ms=0.0,
                citations=CitationValidationResult(),
                error_message=None,
            )

        # 2. Invoke LLM provider with defensive error handling
        try:
            llm_resp: LLMResponse = await self.llm_provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMTimeoutError as t_err:
            logger.error("LLM generation timed out: %s", t_err)
            return AnswerGenerationResult(
                answer="The language model request timed out. Please try again.",
                status=AnswerStatus.TIMEOUT,
                provider=t_err.provider,
                model="unknown",
                error_message=str(t_err),
            )
        except LLMRateLimitError as r_err:
            logger.warning("LLM rate limit reached: %s", r_err)
            return AnswerGenerationResult(
                answer="Language model rate limit reached. Please wait a moment and try again.",
                status=AnswerStatus.RATE_LIMITED,
                provider=r_err.provider,
                model="unknown",
                error_message=str(r_err),
            )
        except LLMAuthenticationError as a_err:
            logger.error("LLM authentication failure: %s", a_err)
            return AnswerGenerationResult(
                answer="Language model authentication failed. Please verify API credentials in configuration.",
                status=AnswerStatus.PROVIDER_ERROR,
                provider=a_err.provider,
                model="unknown",
                error_message="Authentication failure with LLM provider.",
            )
        except LLMProviderError as p_err:
            logger.error("LLM provider error: %s", p_err)
            return AnswerGenerationResult(
                answer="Failed to generate answer due to an upstream language model error.",
                status=AnswerStatus.PROVIDER_ERROR,
                provider=p_err.provider,
                model="unknown",
                error_message=str(p_err),
            )
        except Exception as exc:
            logger.exception("Unexpected error during answer generation: %s", exc)
            return AnswerGenerationResult(
                answer="An unexpected error occurred during answer generation.",
                status=AnswerStatus.PROVIDER_ERROR,
                provider="unknown",
                model="unknown",
                error_message=str(exc),
            )

        # 3. Validate citations in generated text
        citation_result = self.citation_validator.validate_citations(
            answer_text=llm_resp.content,
            sources=sources_list,
        )

        # 4. Determine status: check if model reported insufficient evidence
        status = AnswerStatus.ANSWERED
        lower_ans = llm_resp.content.lower()
        if (
            "not contain sufficient information" in lower_ans
            or "does not contain sufficient information" in lower_ans
            or "insufficient information to answer" in lower_ans
        ):
            status = AnswerStatus.INSUFFICIENT_EVIDENCE

        return AnswerGenerationResult(
            answer=llm_resp.content,
            status=status,
            provider=llm_resp.provider,
            model=llm_resp.model,
            usage=llm_resp.usage,
            latency_ms=llm_resp.latency_ms,
            citations=citation_result,
            error_message=None,
        )
