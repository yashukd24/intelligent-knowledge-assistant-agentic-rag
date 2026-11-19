"""
tests/test_anthropic_provider.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for AnthropicProvider using mocks.
Verifies zero real API calls, proper exception mapping, parameter
forwarding, and token usage extraction.
─────────────────────────────────────────────────────────────
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.config import Settings
from backend.services.anthropic_provider import AnthropicProvider
from backend.services.llm_provider import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseMalformedError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        anthropic_api_key="sk-ant-test-key-12345",
        anthropic_model="claude-3-5-sonnet-20241022",
        anthropic_max_tokens=2048,
        anthropic_timeout_seconds=30.0,
        anthropic_temperature=0.0,
    )


def test_provider_initialization_with_valid_key(mock_settings):
    provider = AnthropicProvider(settings=mock_settings)
    assert provider.model == "claude-3-5-sonnet-20241022"


def test_provider_initialization_empty_api_key(mock_settings):
    provider = AnthropicProvider(api_key="", settings=mock_settings)
    with pytest.raises(LLMAuthenticationError, match="Anthropic API key is missing or unconfigured"):
        provider._get_client()


def test_provider_initialization_placeholder_key(mock_settings):
    provider = AnthropicProvider(api_key="sk-ant-REPLACE_WITH_YOUR_KEY", settings=mock_settings)
    with pytest.raises(LLMAuthenticationError, match="Anthropic API key is missing or unconfigured"):
        provider._get_client()


@pytest.mark.asyncio
async def test_generate_empty_prompt_raises(mock_settings):
    provider = AnthropicProvider(settings=mock_settings)
    with pytest.raises(LLMProviderError, match="Prompt cannot be empty"):
        await provider.generate(prompt="")


@pytest.mark.asyncio
async def test_generate_success(mock_settings):
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "This is a grounded answer from Claude [S1]."

    mock_usage = MagicMock()
    mock_usage.input_tokens = 120
    mock_usage.output_tokens = 45

    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.usage = mock_usage

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    resp = await provider.generate(
        prompt="User prompt here",
        system_instruction="System instructions here",
        temperature=0.2,
        max_tokens=1000,
    )

    assert resp.content == "This is a grounded answer from Claude [S1]."
    assert resp.model == "claude-3-5-sonnet-20241022"
    assert resp.provider == "anthropic"
    assert resp.usage.prompt_tokens == 120
    assert resp.usage.completion_tokens == 45
    assert resp.usage.total_tokens == 165

    # Verify parameters sent to messages.create
    mock_client.messages.create.assert_awaited_once_with(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.2,
        system="System instructions here",
        messages=[{"role": "user", "content": "User prompt here"}],
    )


@pytest.mark.asyncio
async def test_generate_multiple_text_blocks(mock_settings):
    block1 = MagicMock()
    block1.type = "text"
    block1.text = "Part 1 of the answer. "

    block2 = MagicMock()
    block2.type = "text"
    block2.text = "Part 2 of the answer [S1]."

    mock_response = MagicMock()
    mock_response.content = [block1, block2]
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.usage = MagicMock(input_tokens=50, output_tokens=30)

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)
    resp = await provider.generate(prompt="Tell me about X")

    assert resp.content == "Part 1 of the answer. Part 2 of the answer [S1]."


@pytest.mark.asyncio
async def test_generate_empty_content_raises_malformed(mock_settings):
    mock_response = MagicMock()
    mock_response.content = []
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=0)

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMResponseMalformedError, match="empty response or unexpected content blocks"):
        await provider.generate(prompt="Tell me about X")


@pytest.mark.asyncio
async def test_generate_no_text_blocks_raises_malformed(mock_settings):
    mock_block = MagicMock()
    del mock_block.text  # does not have .text attribute

    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=0)

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMResponseMalformedError, match="empty response or unexpected content blocks"):
        await provider.generate(prompt="Tell me about X")


@pytest.mark.asyncio
async def test_generate_authentication_error(mock_settings):
    import anthropic

    mock_error = anthropic.AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(status_code=401),
        body={"error": {"message": "Invalid API Key"}},
    )

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=mock_error)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMAuthenticationError, match="Anthropic authentication failed"):
        await provider.generate(prompt="Test prompt")


@pytest.mark.asyncio
async def test_generate_rate_limit_error(mock_settings):
    import anthropic

    mock_error = anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={"error": {"message": "Rate limit exceeded"}},
    )

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=mock_error)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMRateLimitError, match="Anthropic rate limit exceeded"):
        await provider.generate(prompt="Test prompt")


@pytest.mark.asyncio
async def test_generate_timeout_error(mock_settings):
    import anthropic

    mock_error = anthropic.APITimeoutError(request=MagicMock())

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=mock_error)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMTimeoutError, match="Anthropic API request timed out"):
        await provider.generate(prompt="Test prompt")


@pytest.mark.asyncio
async def test_generate_service_unavailable_error(mock_settings):
    import anthropic

    mock_error = anthropic.InternalServerError(
        message="Overloaded",
        response=MagicMock(status_code=500),
        body={"error": {"message": "Overloaded"}},
    )

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=mock_error)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMServiceUnavailableError, match="Anthropic service is currently unavailable"):
        await provider.generate(prompt="Test prompt")


@pytest.mark.asyncio
async def test_generate_connection_error(mock_settings):
    import anthropic

    mock_error = anthropic.APIConnectionError(request=MagicMock(), message="Connection refused")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=mock_error)

    provider = AnthropicProvider(client=mock_client, settings=mock_settings)

    with pytest.raises(LLMServiceUnavailableError, match="Anthropic service is currently unavailable"):
        await provider.generate(prompt="Test prompt")
