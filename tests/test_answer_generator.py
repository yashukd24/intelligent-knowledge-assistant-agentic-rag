"""
tests/test_answer_generator.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for AnswerGenerator.
Verifies grounded answer orchestration, empty evidence bypass,
citation validation integration, and LLM error status mapping.
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import pytest

from backend.schemas.rag import SourceReference
from backend.services.answer_generator import (
    AnswerGenerationResult,
    AnswerGenerator,
    AnswerStatus,
    STANDARD_INSUFFICIENT_EVIDENCE_ANSWER,
)
from backend.services.llm_provider import (
    LLMAuthenticationError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseMalformedError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    TokenUsage,
)


def _create_mock_source(source_num: int = 1) -> SourceReference:
    return SourceReference(
        source_id=f"[S{source_num}]",
        document_id=str(uuid4()),
        chunk_id=str(uuid4()),
        filename="doc1.pdf",
        page_number=1,
        chunk_index=source_num - 1,
        similarity_score=0.9,
        char_count=100,
    )


@pytest.mark.asyncio
async def test_generate_answer_success_with_evidence():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock(
        return_value=LLMResponse(
            content="According to the documentation [S1], feature X is supported.",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=35, total_tokens=135),
        )
    )

    generator = AnswerGenerator(llm_provider=mock_provider)
    sources = [_create_mock_source(1)]
    prompt = "Context: [Source 1] Content. Question: What is X?"

    result = await generator.generate_answer(
        prompt=prompt,
        sources=sources,
        has_evidence=True,
    )

    assert result.status == AnswerStatus.ANSWERED
    assert "According to the documentation [S1]" in result.answer
    assert result.model == "claude-3-5-sonnet-20241022"
    assert result.usage.total_tokens == 135
    assert result.citations.valid_citations == ["[S1]"]
    assert result.citations.invalid_citations == []
    assert result.error_message is None

    mock_provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_answer_insufficient_evidence_bypasses_llm():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock()

    generator = AnswerGenerator(llm_provider=mock_provider)

    result = await generator.generate_answer(
        prompt="No documents found.",
        sources=[],
        has_evidence=False,
    )

    # Verification: Provider is NEVER called, saving cost and eliminating hallucination
    mock_provider.generate.assert_not_called()

    assert result.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer == STANDARD_INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations.cited_sources == []
    assert result.usage is None
    assert result.error_message is None


@pytest.mark.asyncio
async def test_generate_answer_timeout_mapped_correctly():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock(
        side_effect=LLMTimeoutError("Request to Claude timed out after 30s", provider="anthropic")
    )

    generator = AnswerGenerator(llm_provider=mock_provider)
    sources = [_create_mock_source(1)]

    result = await generator.generate_answer(
        prompt="Prompt here",
        sources=sources,
        has_evidence=True,
    )

    assert result.status == AnswerStatus.TIMEOUT
    assert "timed out" in result.answer.lower()
    assert result.citations.cited_sources == []
    assert result.error_message is not None
    assert "timed out" in result.error_message.lower()


@pytest.mark.asyncio
async def test_generate_answer_rate_limit_mapped_correctly():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock(
        side_effect=LLMRateLimitError("Anthropic rate limit reached", provider="anthropic")
    )

    generator = AnswerGenerator(llm_provider=mock_provider)
    sources = [_create_mock_source(1)]

    result = await generator.generate_answer(
        prompt="Prompt here",
        sources=sources,
        has_evidence=True,
    )

    assert result.status == AnswerStatus.RATE_LIMITED
    assert "rate limit" in result.answer.lower()
    assert result.error_message == "Anthropic rate limit reached"


@pytest.mark.asyncio
async def test_generate_answer_authentication_error_mapped_to_provider_error():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock(
        side_effect=LLMAuthenticationError("Invalid API key", provider="anthropic")
    )

    generator = AnswerGenerator(llm_provider=mock_provider)
    sources = [_create_mock_source(1)]

    result = await generator.generate_answer(
        prompt="Prompt here",
        sources=sources,
        has_evidence=True,
    )

    assert result.status == AnswerStatus.PROVIDER_ERROR
    assert "authentication failed" in result.answer.lower()
    assert "Authentication failure" in result.error_message


@pytest.mark.asyncio
async def test_generate_answer_service_unavailable_mapped_to_provider_error():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock(
        side_effect=LLMServiceUnavailableError("Claude service is currently overloaded", provider="anthropic")
    )

    generator = AnswerGenerator(llm_provider=mock_provider)
    sources = [_create_mock_source(1)]

    result = await generator.generate_answer(
        prompt="Prompt here",
        sources=sources,
        has_evidence=True,
    )

    assert result.status == AnswerStatus.PROVIDER_ERROR
    assert "error" in result.answer.lower()
    assert "overloaded" in result.error_message.lower()


@pytest.mark.asyncio
async def test_generate_answer_malformed_response_mapped_to_provider_error():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock(
        side_effect=LLMResponseMalformedError("No text blocks found in Claude response", provider="anthropic")
    )

    generator = AnswerGenerator(llm_provider=mock_provider)
    sources = [_create_mock_source(1)]

    result = await generator.generate_answer(
        prompt="Prompt here",
        sources=sources,
        has_evidence=True,
    )

    assert result.status == AnswerStatus.PROVIDER_ERROR
    assert "No text blocks" in result.error_message


@pytest.mark.asyncio
async def test_generate_answer_unexpected_exception_mapped_to_provider_error():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate = AsyncMock(
        side_effect=RuntimeError("Unexpected OS level error")
    )

    generator = AnswerGenerator(llm_provider=mock_provider)
    sources = [_create_mock_source(1)]

    result = await generator.generate_answer(
        prompt="Prompt here",
        sources=sources,
        has_evidence=True,
    )

    assert result.status == AnswerStatus.PROVIDER_ERROR
    assert "unexpected error" in result.answer.lower()
    assert "Unexpected OS level error" in result.error_message
