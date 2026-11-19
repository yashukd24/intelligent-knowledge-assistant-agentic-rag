"""
backend/services/llm_provider.py
─────────────────────────────────────────────────────────────
Abstract interface and data structures for Large Language Model (LLM) providers.
Decouples RAG answer generation from specific vendor SDKs (Anthropic, OpenAI, etc.).
─────────────────────────────────────────────────────────────
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Token consumption metadata returned by the provider."""
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(from_attributes=True)


class LLMResponse(BaseModel):
    """Structured response from an LLM provider."""
    content: str = Field(..., description="Generated text content.")
    provider: str = Field(..., description="Provider name (e.g., 'anthropic').")
    model: str = Field(..., description="Model identifier used for generation.")
    usage: Optional[TokenUsage] = Field(default=None, description="Token usage counts if available.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Inference latency in milliseconds.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider metadata.")

    model_config = ConfigDict(from_attributes=True)


# ── Controlled Provider Exceptions ──────────────────────────────────────────


class LLMProviderError(Exception):
    """Base exception for all LLM provider failures."""
    def __init__(self, message: str, provider: str = "unknown", status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code


class LLMAuthenticationError(LLMProviderError):
    """Raised when provider credentials are missing, expired, or invalid."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when provider rate limits or quota thresholds are exceeded."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when an API call times out."""
    pass


class LLMServiceUnavailableError(LLMProviderError):
    """Raised when provider servers are overloaded, down, or return 500/502/503."""
    pass


class LLMResponseMalformedError(LLMProviderError):
    """Raised when the provider response is empty, unexpected, or unparseable."""
    pass


# ── Abstract LLM Provider Interface ─────────────────────────────────────────


class LLMProvider(ABC):
    """
    Abstract interface for LLM completion providers.
    All implementations must be async and thread-safe.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate a completion asynchronously.

        Args:
            prompt: User prompt content.
            system_instruction: Optional system instruction / persona prefix.
            max_tokens: Maximum output tokens to generate.
            temperature: Sampling temperature in [0.0, 1.0].

        Returns:
            LLMResponse containing generated text and execution metadata.

        Raises:
            LLMAuthenticationError
            LLMRateLimitError
            LLMTimeoutError
            LLMServiceUnavailableError
            LLMResponseMalformedError
            LLMProviderError
        """
        pass
