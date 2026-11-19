"""
backend/config.py
─────────────────────────────────────────────────────────────
Application configuration loaded from environment variables.
Uses pydantic-settings so every value is validated at startup.

Usage:
    from backend.config import get_settings
    settings = get_settings()
    print(settings.backend_port)
─────────────────────────────────────────────────────────────
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings.
    Values are loaded from the .env file, then environment variables,
    with these defaults as fallback.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # Ignore unknown env vars silently
    )

    # ── Project identity ────────────────────────────────────
    app_name: str = "Intelligent Knowledge Assistant"
    app_version: str = "0.1.0"
    app_description: str = "Agentic RAG system using Claude and ChromaDB"

    # ── LLM Provider Selection ──────────────────────────────
    llm_provider: str = Field(
        default="ollama",
        description="Active LLM provider: 'ollama' (default local) or 'anthropic' (cloud)",
    )

    # ── Ollama Local LLM ─────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama daemon",
    )
    ollama_model: str = Field(
        default="llama3:latest",
        description="Local Ollama model name (e.g. llama3:latest)",
    )
    ollama_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=600.0,
        description="Timeout for local Ollama requests in seconds",
    )
    ollama_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for Ollama generation",
    )
    ollama_num_ctx: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Context window in tokens for Ollama (conservative to fit system RAM)",
    )
    ollama_max_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Maximum output tokens for Ollama CPU generation",
    )

    # ── Anthropic / Claude ───────────────────────────────────
    anthropic_api_key: str = Field(
        default="sk-ant-REPLACE_WITH_YOUR_KEY",
        description="Anthropic API key — set in .env before Phase 9",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-5",
        description="Anthropic model name (e.g. claude-sonnet-5)",
    )
    claude_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = Field(default=1024, ge=1, le=8192)
    claude_max_tokens: int = 4096
    anthropic_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    claude_temperature: float = 0.1
    anthropic_temperature: float = 0.1

    # ── Backend server ───────────────────────────────────────
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    backend_reload: bool = True

    # ── Frontend ─────────────────────────────────────────────
    frontend_port: int = 8501

    # ── Database ─────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/knowledge_assistant.db"

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chromadb"
    chroma_collection_name: str = "knowledge_base"

    # ── Embedding model ──────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_dimension: int = 384
    embedding_batch_size: int = 1
    embedding_normalize: bool = True
    embedding_cache_dir: Optional[str] = None

    # ── Document processing ──────────────────────────────────
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50
    supported_extensions: str = ".pdf,.docx,.txt"

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ────────────────────────────────────────────
    top_k_results: int = 5
    similarity_threshold: float = 0.3

    # ── Agent ────────────────────────────────────────────────
    agent_max_iterations: int = 5
    agent_refinement_rounds: int = 2

    # ── Logging ──────────────────────────────────────────────
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # ── Evaluation ───────────────────────────────────────────
    eval_questions_path: str = "./data/eval_questions/questions.json"

    # ── Computed properties ──────────────────────────────────
    @property
    def supported_extensions_list(self) -> list[str]:
        """Return supported extensions as a list: ['.pdf', '.docx', '.txt']"""
        return [ext.strip() for ext in self.supported_extensions.split(",")]

    @property
    def upload_dir_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def log_dir_path(self) -> Path:
        return Path(self.log_dir)

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def effective_embedding_cache_dir(self) -> Optional[str]:
        """
        Return the configured embedding cache directory, falling back
        to the HF_HOME environment variable, or None (default HF cache).
        """
        return self.embedding_cache_dir or os.environ.get("HF_HOME")

    @field_validator("claude_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("claude_temperature must be between 0.0 and 1.0")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        # overlap must be less than chunk_size — checked post-init
        if v < 0:
            raise ValueError("chunk_overlap must be >= 0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.
    Cached after first call — safe to call anywhere.
    """
    return Settings()
