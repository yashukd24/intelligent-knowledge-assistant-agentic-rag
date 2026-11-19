"""backend/routers/__init__.py"""
from backend.routers.collections import router as collections_router
from backend.routers.documents import collection_docs_router, docs_router
from backend.routers.health import router as health_router

__all__ = [
    "health_router",
    "collections_router",
    "collection_docs_router",
    "docs_router",
]
