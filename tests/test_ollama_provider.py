"""
tests/test_ollama_provider.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for OllamaProvider using HTTP mocks.
Verifies zero real API calls, proper exception mapping, parameter
forwarding (options, num_ctx, num_predict), token usage extraction,
and dynamic provider selection in get_llm_provider.
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from backend.config import Settings
from backend.routers.rag_answer import get_llm_provider
from backend.services.anthropic_provider import AnthropicProvider
from backend.services.llm_provider import (
    LLMProviderError,
    LLMResponseMalformedError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from backend.services.ollama_provider import OllamaProvider


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3:latest",
        ollama_num_ctx=2048,
        ollama_max_tokens=512,
        ollama_timeout_seconds=120.0,
        ollama_temperature=0.1,
    )


def test_ollama_provider_initialization_defaults(mock_settings):
    provider = OllamaProvider(settings=mock_settings)
    assert provider.base_url == "http://localhost:11434"
    assert provider.model == "llama3:latest"
    assert provider.num_ctx == 2048
    assert provider.max_tokens == 512
    assert provider.timeout_seconds == 120.0
    assert provider.temperature == 0.1


def test_ollama_provider_initialization_custom_overrides(mock_settings):
    provider = OllamaProvider(
        base_url="http://192.168.1.100:11434/",
        model="custom-llama:latest",
        num_ctx=1024,
        max_tokens=256,
        timeout_seconds=60.0,
        temperature=0.5,
        settings=mock_settings,
    )
    assert provider.base_url == "http://192.168.1.100:11434"
    assert provider.model == "custom-llama:latest"
    assert provider.num_ctx == 1024
    assert provider.max_tokens == 256
    assert provider.timeout_seconds == 60.0
    assert provider.temperature == 0.5


@pytest.mark.asyncio
async def test_ollama_generate_empty_prompt_raises(mock_settings):
    provider = OllamaProvider(settings=mock_settings)
    with pytest.raises(LLMProviderError, match="Prompt cannot be empty"):
        await provider.generate(prompt="")


@pytest.mark.asyncio
async def test_ollama_generate_success(mock_settings):
    mock_response_data = {
        "model": "llama3:latest",
        "created_at": "2026-11-10T09:00:00Z",
        "message": {
            "role": "assistant",
            "content": "According to the documentation [S1], feature X is supported.",
        },
        "done": True,
        "total_duration": 1500000000,  # 1.5s in ns -> 1500.0 ms
        "prompt_eval_count": 120,
        "eval_count": 35,
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)

    provider = OllamaProvider(client=mock_client, settings=mock_settings)

    resp = await provider.generate(
        prompt="What is feature X?",
        system_instruction="You are a helpful assistant.",
        max_tokens=400,
        temperature=0.2,
    )

    assert resp.content == "According to the documentation [S1], feature X is supported."
    assert resp.provider == "ollama"
    assert resp.model == "llama3:latest"
    assert resp.usage.prompt_tokens == 120
    assert resp.usage.completion_tokens == 35
    assert resp.usage.total_tokens == 155
    assert resp.latency_ms == 1500.0

    # Verify payload dispatched to Ollama
    mock_client.post.assert_awaited_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3:latest",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is feature X?"},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 400,
                "num_ctx": 2048,
            },
        },
    )


@pytest.mark.asyncio
async def test_ollama_generate_connection_error(mock_settings):
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    provider = OllamaProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMServiceUnavailableError, match="Ollama service is not reachable"):
        await provider.generate(prompt="Hello")


@pytest.mark.asyncio
async def test_ollama_generate_timeout_error(mock_settings):
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("Read timed out"))

    provider = OllamaProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMTimeoutError, match="timed out"):
        await provider.generate(prompt="Hello")


@pytest.mark.asyncio
async def test_ollama_generate_model_not_found_404(mock_settings):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.text = "model 'llama3:latest' not found"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)

    provider = OllamaProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMProviderError, match="not found"):
        await provider.generate(prompt="Hello")


@pytest.mark.asyncio
async def test_ollama_generate_error_in_json_body(mock_settings):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"error": "model 'llama3:latest' not found"}

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)

    provider = OllamaProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMProviderError, match="not found"):
        await provider.generate(prompt="Hello")


@pytest.mark.asyncio
async def test_ollama_generate_server_error_500(mock_settings):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)

    provider = OllamaProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMServiceUnavailableError, match="Ollama server error"):
        await provider.generate(prompt="Hello")


@pytest.mark.asyncio
async def test_ollama_generate_empty_content_raises_malformed(mock_settings):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "llama3:latest",
        "message": {"role": "assistant", "content": "   "},
        "done": True,
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)

    provider = OllamaProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMResponseMalformedError, match="empty response content"):
        await provider.generate(prompt="Hello")


def test_provider_selection_ollama(mock_settings):
    mock_settings.llm_provider = "ollama"
    provider = get_llm_provider(settings=mock_settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3:latest"


def test_provider_selection_anthropic():
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test-key",
        anthropic_model="claude-sonnet-5",
    )
    provider = get_llm_provider(settings=settings)
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-5"


def test_provider_selection_default_local_without_anthropic_key():
    # Demonstrates Anthropic is NEVER initialized in local mode
    # and no ANTHROPIC_API_KEY is required
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        llm_provider="ollama",
        anthropic_api_key="",  # empty key
    )
    provider = get_llm_provider(settings=settings)
    assert isinstance(provider, OllamaProvider)
