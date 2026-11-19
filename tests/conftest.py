"""
tests/conftest.py
─────────────────────────────────────────────────────────────
Shared pytest fixtures for the entire test suite.

Design decisions for pytest 8.4 + pytest-asyncio 0.24 + Windows:
  - asyncio_mode = "auto" (set in pyproject.toml)
  - ALL fixtures are function-scoped (default) to avoid event-loop
    sharing issues between session-scoped async fixtures and tests
  - The FastAPI lifespan is bypassed in tests: DB init is handled
    directly in the fixture, logging is kept at WARNING level
  - Each test gets its own fresh in-memory SQLite database

Fixtures provided:
  - test_settings : Session-scoped Settings (no async — always safe)
  - client        : Function-scoped async HTTPX client (one per test)
─────────────────────────────────────────────────────────────
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import Settings, get_settings
from backend.database import get_db, setup_sqlite_foreign_keys
from backend.models import Base, Collection, Document
from backend.routers.collections import get_storage_service as get_col_storage
from backend.routers.documents import (
    get_document_processor,
    get_storage_service as get_doc_storage,
)
from backend.services.document_processor import DocumentProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.file_storage import FileStorageService

# Silence noisy loggers during tests
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("backend").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def test_upload_dir(tmp_path) -> Path:
    """Isolated temporary upload directory per test."""
    upload_path = tmp_path / "uploads"
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


@pytest.fixture
def test_chroma_dir(tmp_path) -> Path:
    """Isolated temporary ChromaDB persistence directory per test."""
    chroma_path = tmp_path / "chromadb"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chroma_path


@pytest.fixture
def test_settings(test_upload_dir, test_chroma_dir) -> Settings:
    """Function-scoped settings — in-memory DB and isolated temporary storage/chroma directories."""
    return Settings(
        database_url=TEST_DB_URL,
        anthropic_api_key="sk-ant-TEST_KEY",
        log_level="WARNING",
        backend_reload=False,
        upload_dir=str(test_upload_dir),
        chroma_persist_dir=str(test_chroma_dir),
    )


@pytest.fixture
def test_vector_store(test_chroma_dir) -> "VectorStoreService":
    """Provides an isolated VectorStoreService pointing to temporary test directory."""
    from backend.services.vector_store import VectorStoreService
    return VectorStoreService(persist_dir=str(test_chroma_dir))


@pytest_asyncio.fixture
async def db_session(test_settings) -> AsyncGenerator[AsyncSession, None]:
    """Function-scoped direct database session for testing constraints and transactions."""
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        future=True,
    )
    setup_sqlite_foreign_keys(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(test_settings, test_upload_dir, test_chroma_dir) -> AsyncClient:
    """
    Function-scoped async test client with isolated in-memory DB and temp storage.
    """
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        future=True,
    )
    setup_sqlite_foreign_keys(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    from backend.main import create_app
    from backend.routers.collections import get_vector_store as get_col_vector_store
    from backend.routers.documents import (
        get_document_processor,
        get_storage_service as get_doc_storage,
        get_vector_store,
    )
    from backend.routers.rag import get_rag_service
    from backend.routers.rag_answer import (
        get_answer_generator,
        get_llm_provider,
        get_rag_answer_service,
    )
    from backend.routers.retrieval import (
        get_retrieval_service,
        get_vector_store as get_retrieval_vector_store,
    )
    from backend.services.answer_generator import AnswerGenerator
    from backend.services.llm_provider import LLMProvider, LLMResponse, TokenUsage
    from backend.services.rag_answer_service import RAGAnswerService
    from backend.services.rag_service import RAGService
    from backend.services.retrieval_service import RetrievalService
    from backend.services.vector_store import VectorStoreService

    app = create_app()
    app.state.session_factory = factory

    # In-memory settings and storage service
    storage_svc = FileStorageService(upload_dir=test_upload_dir)
    vector_svc = VectorStoreService(persist_dir=str(test_chroma_dir))

    class _MockTestSentenceTransformer:
        """Deterministic mock SentenceTransformer returning valid 384-d float32 unit vectors."""
        def __init__(self, dimension: int = 384):
            self._dim = dimension

        def get_sentence_embedding_dimension(self) -> int:
            return self._dim

        def encode(
            self,
            sentences,
            batch_size: int = 1,
            show_progress_bar: bool = False,
            normalize_embeddings: bool = True,
            convert_to_numpy: bool = True,
        ):
            import numpy as np
            if isinstance(sentences, str):
                sentences = [sentences]
            vectors = []
            for s in sentences:
                seed = abs(hash(s)) % (2**31)
                rng = np.random.RandomState(seed)
                vec = rng.randn(self._dim).astype(np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                vectors.append(vec)
            return np.array(vectors)

    mock_embed_svc = EmbeddingService(
        model=_MockTestSentenceTransformer(dimension=384),
        settings=test_settings,
    )
    doc_processor = DocumentProcessor(
        storage_service=storage_svc,
        embedding_service=mock_embed_svc,
        vector_store=vector_svc,
    )
    retrieval_svc = RetrievalService(
        embedding_service=mock_embed_svc,
        settings=test_settings,
        vector_store=vector_svc,
    )
    rag_svc = RAGService(retrieval_service=retrieval_svc, settings=test_settings)

    class _MockTestLLMProvider(LLMProvider):
        async def generate(
            self,
            prompt: str,
            system_instruction: str | None = None,
            max_tokens: int | None = None,
            temperature: float | None = None,
        ) -> LLMResponse:
            return LLMResponse(
                content="This is a test answer [S1].",
                provider="mock",
                model="claude-test-mock",
                usage=TokenUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23),
            )

    test_llm_provider = _MockTestLLMProvider()
    test_answer_gen = AnswerGenerator(llm_provider=test_llm_provider)
    rag_answer_svc = RAGAnswerService(rag_service=rag_svc, answer_generator=test_answer_gen)

    from backend.routers.agent import get_agent_controller
    from backend.services.agent_controller import AgentController
    agent_ctrl = AgentController(
        rag_service=rag_svc,
        answer_generator=test_answer_gen,
        settings=test_settings,
    )

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_col_storage] = lambda: storage_svc
    app.dependency_overrides[get_doc_storage] = lambda: storage_svc
    app.dependency_overrides[get_document_processor] = lambda: doc_processor
    app.dependency_overrides[get_vector_store] = lambda: vector_svc
    app.dependency_overrides[get_col_vector_store] = lambda: vector_svc
    app.dependency_overrides[get_retrieval_vector_store] = lambda: vector_svc
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_svc
    app.dependency_overrides[get_rag_service] = lambda: rag_svc
    app.dependency_overrides[get_llm_provider] = lambda: test_llm_provider
    app.dependency_overrides[get_answer_generator] = lambda: test_answer_gen
    app.dependency_overrides[get_rag_answer_service] = lambda: rag_answer_svc
    app.dependency_overrides[get_agent_controller] = lambda: agent_ctrl

    async def _override_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()
