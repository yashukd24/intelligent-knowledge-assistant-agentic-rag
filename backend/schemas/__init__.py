"""backend/schemas/__init__.py"""
from backend.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    ProcessedDocument,
    TextPage,
)
from backend.schemas.embedding import EmbeddedChunk
from backend.schemas.collection import (
    CollectionCreate,
    CollectionListResponse,
    CollectionResponse,
    CollectionUpdate,
)
from backend.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
)
from backend.schemas.health import ComponentHealth, HealthResponse, ServiceStatus

__all__ = [
    "HealthResponse",
    "ComponentHealth",
    "ServiceStatus",
    "CollectionCreate",
    "CollectionUpdate",
    "CollectionResponse",
    "CollectionListResponse",
    "DocumentStatus",
    "DocumentResponse",
    "DocumentListResponse",
    "TextPage",
    "ChunkMetadata",
    "DocumentChunk",
    "ProcessedDocument",
    "EmbeddedChunk",
]

# Rebuild ProcessedDocument so that the forward-referenced EmbeddedChunk is resolved
ProcessedDocument.model_rebuild()
