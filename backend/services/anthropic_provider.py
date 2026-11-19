"""
backend/services/anthropic_provider.py
─────────────────────────────────────────────────────────────
Anthropic Claude implementation of the LLMProvider interface.
Uses the official AsyncAnthropic client with strict error handling,
credential masking, and timeout controls.
─────────────────────────────────────────────────────────────
"""

import logging
import time
from typing import Any, Optional

import anthropic
from anthropic import AsyncAnthropic

from backend.config import Settings, get_settings
from backend.services.llm_provider import (
    LLMAuthenticationError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseMalformedError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    TokenUsage,
)

logger = logging.getLogger(__name__)

PLACEHOLDER_KEY = "sk-ant-REPLACE_WITH_YOUR_KEY"


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude LLM provider using the asynchronous AsyncAnthropic SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        temperature: Optional[float] = None,
        client: Optional[AsyncAnthropic] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the AnthropicProvider.

        Args:
            api_key: Optional API key. Defaults to settings.anthropic_api_key.
            model: Optional model identifier. Defaults to settings.anthropic_model.
            max_tokens: Optional default max output tokens. Defaults to settings.anthropic_max_tokens.
            timeout_seconds: Optional timeout in seconds. Defaults to settings.anthropic_timeout_seconds.
            temperature: Optional default temperature. Defaults to settings.anthropic_temperature.
            client: Optional pre-configured AsyncAnthropic client (useful for unit tests/mocking).
            settings: Optional Settings singleton.
        """
        self.settings = settings or get_settings()

        # Resolve configuration with fallbacks
        self._api_key = api_key if api_key is not None else self.settings.anthropic_api_key
        self.model = model or getattr(self.settings, "anthropic_model", None) or self.settings.claude_model
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(self.settings, "anthropic_max_tokens", 1024)
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings, "anthropic_timeout_seconds", 30.0)
        )
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(self.settings, "anthropic_temperature", 0.1)
        )

        # Allow injected client for testing; otherwise construct on demand
        self._injected_client = client

    def _get_client(self) -> AsyncAnthropic:
        """Instantiate or return the AsyncAnthropic client, validating the API key."""
        if self._injected_client is not None:
            return self._injected_client

        if not self._api_key or not self._api_key.strip() or self._api_key == PLACEHOLDER_KEY:
            raise LLMAuthenticationError(
                "Anthropic API key is missing or unconfigured. Please configure ANTHROPIC_API_KEY in your .env file.",
                provider="anthropic",
            )

        return AsyncAnthropic(
            api_key=self._api_key,
            timeout=float(self.timeout_seconds),
        )

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate completion from Claude using Messages API asynchronously.

        Args:
            prompt: User message prompt string.
            system_instruction: Optional system instruction prompt.
            max_tokens: Override for maximum output tokens.
            temperature: Override for sampling temperature.

        Returns:
            Structured LLMResponse model.

        Raises:
            LLMAuthenticationError: When API key is invalid or rejected.
            LLMRateLimitError: When Anthropic rate limits or quotas are hit.
            LLMTimeoutError: When the network call times out.
            LLMServiceUnavailableError: When Anthropic servers return 5xx errors.
            LLMResponseMalformedError: When response contains no text.
            LLMProviderError: On other API errors.
        """
        if not prompt or not prompt.strip():
            raise LLMProviderError("Prompt cannot be empty.", provider="anthropic")

        client = self._get_client()

        req_model = self.model
        req_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        req_temp = temperature if temperature is not None else self.temperature

        messages_payload = [{"role": "user", "content": prompt}]

        kwargs: dict[str, Any] = {
            "model": req_model,
            "max_tokens": req_max_tokens,
            "temperature": req_temp,
            "messages": messages_payload,
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        t0 = time.perf_counter()

        try:
            logger.info("Sending completion request to Anthropic Claude model: %s", req_model)
            response = await client.messages.create(**kwargs)
        except anthropic.AuthenticationError as auth_err:
            logger.error("Anthropic authentication failed: %s", auth_err.message)
            raise LLMAuthenticationError(
                f"Anthropic authentication failed: {auth_err.message}",
                provider="anthropic",
                status_code=auth_err.status_code,
            ) from auth_err
        except anthropic.RateLimitError as rate_err:
            logger.warning("Anthropic rate limit reached: %s", rate_err.message)
            raise LLMRateLimitError(
                f"Anthropic rate limit exceeded: {rate_err.message}",
                provider="anthropic",
                status_code=rate_err.status_code,
            ) from rate_err
        except anthropic.APITimeoutError as timeout_err:
            logger.error("Anthropic request timed out after %.1fs", self.timeout_seconds)
            raise LLMTimeoutError(
                f"Anthropic API request timed out after {self.timeout_seconds}s.",
                provider="anthropic",
            ) from timeout_err
        except (anthropic.InternalServerError, anthropic.APIConnectionError) as conn_err:
            logger.error("Anthropic service error or connection failure: %s", conn_err)
            raise LLMServiceUnavailableError(
                f"Anthropic service is currently unavailable: {conn_err}",
                provider="anthropic",
            ) from conn_err
        except anthropic.APIStatusError as status_err:
            logger.error("Anthropic API returned status %s: %s", status_err.status_code, status_err.message)
            raise LLMProviderError(
                f"Anthropic API error: {status_err.message}",
                provider="anthropic",
                status_code=status_err.status_code,
            ) from status_err
        except Exception as exc:
            logger.exception("Unexpected error communicating with Anthropic API: %s", exc)
            raise LLMProviderError(f"Unexpected provider error: {str(exc)}", provider="anthropic") from exc
        finally:
            t1 = time.perf_counter()

        latency_ms = round((t1 - t0) * 1000.0, 2)

        # 4. Extract generated text from content blocks
        text_parts: list[str] = []
        if response and response.content:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    text_parts.append(block.text)

        if not text_parts:
            logger.error("Anthropic response contained no valid text blocks: %s", response)
            raise LLMResponseMalformedError(
                "Anthropic returned an empty response or unexpected content blocks.",
                provider="anthropic",
            )

        generated_text = "".join(text_parts).strip()

        # 5. Extract token usage metadata
        usage_obj: Optional[TokenUsage] = None
        if hasattr(response, "usage") and response.usage:
            prompt_toks = getattr(response.usage, "input_tokens", None)
            comp_toks = getattr(response.usage, "output_tokens", None)
            total_toks = (prompt_toks + comp_toks) if (prompt_toks is not None and comp_toks is not None) else None
            usage_obj = TokenUsage(
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                total_tokens=total_toks,
            )

        return LLMResponse(
            content=generated_text,
            provider="anthropic",
            model=getattr(response, "model", req_model),
            usage=usage_obj,
            latency_ms=latency_ms,
            metadata={
                "id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )
