"""
backend/database.py
─────────────────────────────────────────────────────────────
SQLAlchemy 2.0 async database setup.

Provides:
  - engine        : AsyncEngine connected to SQLite via aiosqlite
  - AsyncSessionLocal : async session factory
  - get_db()      : FastAPI dependency that yields a session per request
  - init_db()     : creates all tables on startup (called from lifespan)
─────────────────────────────────────────────────────────────
"""

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import get_settings
from backend.models import Base, Collection, Document

logger = logging.getLogger(__name__)


def setup_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """
    Ensure SQLite enforces foreign key constraints.
    SQLite ignores foreign keys by default unless PRAGMA foreign_keys=ON is executed
    on every new database connection.
    """
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _build_engine() -> AsyncEngine:
    """
    Create the async SQLAlchemy engine.
    Ensures the SQLite database directory exists before connecting.
    """
    settings = get_settings()
    db_url = settings.database_url

    # For SQLite, ensure the parent directory exists
    if db_url.startswith("sqlite"):
        # Extract file path from URL like sqlite+aiosqlite:///./data/file.db
        db_path_str = db_url.split("///")[-1]
        db_path = Path(db_path_str)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("SQLite database path: %s", db_path.resolve())

    engine = create_async_engine(
        db_url,
        echo=False,            # Set True to log all SQL statements (debug only)
        future=True,
        connect_args={"check_same_thread": False},  # Required for SQLite
    )
    if "sqlite" in db_url:
        setup_sqlite_foreign_keys(engine)
    return engine


# Module-level singletons
engine: AsyncEngine = _build_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # Keep objects usable after commit
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """
    Create all database tables defined in ORM models and migrate columns idempotently.
    Called once during application startup (lifespan).
    Safe to call on an already-initialized database — it is idempotent.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Non-destructive migration for newly added metadata columns in existing SQLite tables
        def _migrate_columns(sync_conn):
            from sqlalchemy import text
            try:
                res = sync_conn.execute(text("PRAGMA table_info(documents)")).fetchall()
                existing_cols = {row[1] for row in res}
                cols_to_add = [
                    ("category", "VARCHAR(100)"),
                    ("department", "VARCHAR(100)"),
                    ("academic_year", "VARCHAR(50)"),
                    ("effective_date", "VARCHAR(50)"),
                    ("expiry_date", "VARCHAR(50)"),
                    ("section_heading", "VARCHAR(255)"),
                ]
                for col_name, col_type in cols_to_add:
                    if col_name not in existing_cols:
                        sync_conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}"))
                        logger.info("Migrated documents table: added column %s", col_name)
            except Exception as e:
                logger.warning("Column migration check passed or not applicable: %s", e)

        await conn.run_sync(_migrate_columns)
    logger.info("Database tables initialized successfully.")


async def close_db() -> None:
    """
    Dispose of the engine connection pool.
    Called during application shutdown (lifespan).
    """
    await engine.dispose()
    logger.info("Database engine disposed.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yields one async database session per request.
    The session is committed on success, rolled back on exception,
    and always closed when the request completes.

    Usage in a router:
        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
