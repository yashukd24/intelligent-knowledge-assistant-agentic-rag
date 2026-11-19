"""
backend/services/ollama_provider.py
─────────────────────────────────────────────────────────────
Local Ollama LLM provider implementing the LLMProvider interface.
Communicates with the local Ollama daemon asynchronously over HTTP,
respects conservative context and token budgets, maps errors into
the standardized LLMProvider exception hierarchy, and supports
dependency-injected HTTP clients for testing.
─────────────────────────────────────────────────────────────
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from backend.config import Settings, get_settings
from backend.services.llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMResponseMalformedError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Local LLM completion provider backed by the Ollama HTTP API.
    Designed for zero external network calls, zero API keys, and
    conservative RAM utilization.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        max_tokens: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the OllamaProvider.

        Args:
            base_url: Ollama daemon URL (e.g. http://localhost:11434).
            model: Ollama model identifier (e.g. llama3:latest).
            timeout_seconds: Timeout for inference HTTP requests.
            temperature: Sampling temperature for generation.
            num_ctx: Context window size in tokens (limits KV cache memory).
            max_tokens: Max tokens to generate per completion.
            client: Optional injected AsyncClient (for mocking in unit tests).
            settings: Optional Settings singleton.
        """
        self.settings = settings or get_settings()

        self.base_url = (
            base_url
            if base_url is not None
            else getattr(self.settings, "ollama_base_url", "http://localhost:11434")
        ).rstrip("/")

        self.model = (
            model
            if model is not None
            else getattr(self.settings, "ollama_model", "llama3:latest")
        )

        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings, "ollama_timeout_seconds", 120.0)
        )

        self.temperature = (
            temperature
            if temperature is not None
            else getattr(self.settings, "ollama_temperature", 0.1)
        )

        self.num_ctx = (
            num_ctx
            if num_ctx is not None
            else getattr(self.settings, "ollama_num_ctx", 2048)
        )

        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(self.settings, "ollama_max_tokens", 512)
        )

        self._injected_client = client

    def _get_client(self) -> httpx.AsyncClient:
        """Return the injected client or instantiate an AsyncClient with timeout."""
        if self._injected_client is not None:
            return self._injected_client
        return httpx.AsyncClient(timeout=float(self.timeout_seconds))

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the local Ollama model asynchronously.

        Args:
            prompt: User question or assembled prompt.
            system_instruction: Optional system instruction persona.
            max_tokens: Override for maximum generated tokens.
            temperature: Override for sampling temperature.

        Returns:
            Structured LLMResponse containing generated content and metadata.

        Raises:
            LLMProviderError: On empty prompt, model not found, or API error.
            LLMServiceUnavailableError: If Ollama daemon is offline or returns 5xx.
            LLMTimeoutError: If inference exceeds timeout limit.
            LLMResponseMalformedError: If Ollama response is empty or unparseable.
        """
        if not prompt or not prompt.strip():
            raise LLMProviderError("Prompt cannot be empty.", provider="ollama")

        req_model = self.model
        req_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        req_temp = temperature if temperature is not None else self.temperature

        messages: List[Dict[str, str]] = []
        if system_instruction and system_instruction.strip():
            messages.append({"role": "system", "content": system_instruction.strip()})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {
            "temperature": req_temp,
            "num_predict": req_max_tokens,
            "num_ctx": self.num_ctx,
        }

        payload: Dict[str, Any] = {
            "model": req_model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        url = f"{self.base_url}/api/chat"
        client = self._get_client()
        owns_client = self._injected_client is None

        t0 = time.perf_counter()

        try:
            logger.info("Dispatching local inference to Ollama (model: %s, url: %s)", req_model, url)
            response = await client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as conn_err:
            logger.error("Failed to connect to local Ollama daemon at %s: %s", self.base_url, conn_err)
            raise LLMServiceUnavailableError(
                f"Ollama service is not reachable at {self.base_url}. Ensure the Ollama daemon is running.",
                provider="ollama",
            ) from conn_err
        except (httpx.ReadTimeout, httpx.TimeoutException) as time_err:
            logger.error("Ollama inference timed out after %.1fs: %s", self.timeout_seconds, time_err)
            raise LLMTimeoutError(
                f"Ollama API request timed out after {self.timeout_seconds}s.",
                provider="ollama",
            ) from time_err
        except Exception as exc:
            logger.exception("Unexpected error communicating with Ollama: %s", exc)
            raise LLMProviderError(f"Unexpected Ollama communication error: {exc}", provider="ollama") from exc
        finally:
            t1 = time.perf_counter()
            if owns_client:
                await client.aclose()

        latency_ms = round((t1 - t0) * 1000.0, 2)

        # Handle HTTP status codes
        if response.status_code == 404:
            raise LLMProviderError(
                f"Ollama model '{req_model}' not found on server {self.base_url}. "
                "Ensure the model is pulled using 'ollama pull " f"{req_model}'.",
                provider="ollama",
                status_code=404,
            )
        elif response.status_code >= 500:
            raise LLMServiceUnavailableError(
                f"Ollama server error ({response.status_code}): {response.text}",
                provider="ollama",
                status_code=response.status_code,
            )
        elif response.status_code != 200:
            raise LLMProviderError(
                f"Ollama returned unexpected status code {response.status_code}: {response.text}",
                provider="ollama",
                status_code=response.status_code,
            )

        # Parse JSON response
        try:
            data = response.json()
        except Exception as json_err:
            raise LLMResponseMalformedError(
                f"Failed to parse Ollama response as JSON: {json_err}",
                provider="ollama",
            ) from json_err

        if "error" in data:
            err_msg = data["error"]
            if "not found" in err_msg.lower():
                raise LLMProviderError(
                    f"Ollama model error: {err_msg}",
                    provider="ollama",
                    status_code=404,
                )
            raise LLMProviderError(f"Ollama returned error: {err_msg}", provider="ollama")

        # Extract generated content
        message_dict = data.get("message") or {}
        content = message_dict.get("content") or ""

        if not content.strip():
            logger.error("Ollama returned empty message content: %s", data)
            raise LLMResponseMalformedError(
                "Ollama returned empty response content.",
                provider="ollama",
            )

        # Extract token usage if available
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = None
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        usage = None
        if total_tokens is not None:
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        # Use Ollama's measured total_duration if provided
        total_duration_ns = data.get("total_duration")
        if total_duration_ns:
            latency_ms = round(total_duration_ns / 1_000_000.0, 2)

        metadata = {
            "done": data.get("done", True),
            "done_reason": data.get("done_reason"),
            "load_duration_ms": round(data.get("load_duration", 0) / 1_000_000.0, 2),
            "prompt_eval_duration_ms": round(data.get("prompt_eval_duration", 0) / 1_000_000.0, 2),
            "eval_duration_ms": round(data.get("eval_duration", 0) / 1_000_000.0, 2),
        }

        return LLMResponse(
            content=content,
            provider="ollama",
            model=req_model,
            usage=usage,
            latency_ms=latency_ms,
            metadata=metadata,
        )
