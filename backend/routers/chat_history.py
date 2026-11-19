"""
backend/routers/chat_history.py
─────────────────────────────────────────────────────────────
FastAPI REST router for Chat History, Session Storage, and Run Logging (Phase 17 & 18).
─────────────────────────────────────────────────────────────
"""

import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.chat_history import ChatMessageRecord
from backend.models.collection import Collection
from backend.schemas.chat_history import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatHistoryResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/collections", tags=["Chat History"])


@router.get(
    "/{collection_id}/chat/history",
    response_model=ChatHistoryResponse,
    summary="Fetch chat history for a collection",
)
async def get_chat_history(
    collection_id: UUID,
    session_id: Optional[str] = Query(None, description="Optional session filter"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    col_str = str(collection_id)
    # Verify collection exists
    col_query = select(Collection).where(Collection.id == col_str)
    res = await db.execute(col_query)
    if not res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found.",
        )

    stmt = (
        select(ChatMessageRecord)
        .where(ChatMessageRecord.collection_id == col_str)
    )
    if session_id:
        stmt = stmt.where(ChatMessageRecord.session_id == session_id)
    stmt = stmt.order_by(ChatMessageRecord.created_at.asc()).limit(limit)

    records = (await db.execute(stmt)).scalars().all()

    messages = []
    for r in records:
        meta = None
        if r.metadata_json:
            try:
                meta = json.loads(r.metadata_json)
            except Exception:
                meta = None
        messages.append(
            ChatMessageResponse(
                id=r.id,
                collection_id=r.collection_id,
                session_id=r.session_id,
                role=r.role,
                content=r.content,
                metadata=meta,
                created_at=r.created_at,
            )
        )

    return ChatHistoryResponse(
        collection_id=col_str,
        session_id=session_id,
        total=len(messages),
        messages=messages,
    )


@router.post(
    "/{collection_id}/chat/history",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a chat message into persistent history",
)
async def save_chat_message(
    collection_id: UUID,
    payload: ChatMessageCreate,
    db: AsyncSession = Depends(get_db),
) -> ChatMessageResponse:
    col_str = str(collection_id)
    col_query = select(Collection).where(Collection.id == col_str)
    res = await db.execute(col_query)
    if not res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found.",
        )

    meta_str = json.dumps(payload.metadata) if payload.metadata else None
    record = ChatMessageRecord(
        collection_id=col_str,
        session_id=payload.session_id,
        role=payload.role,
        content=payload.content,
        metadata_json=meta_str,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    meta = None
    if record.metadata_json:
        try:
            meta = json.loads(record.metadata_json)
        except Exception:
            pass

    return ChatMessageResponse(
        id=record.id,
        collection_id=record.collection_id,
        session_id=record.session_id,
        role=record.role,
        content=record.content,
        metadata=meta,
        created_at=record.created_at,
    )


@router.delete(
    "/{collection_id}/chat/history",
    status_code=status.HTTP_200_OK,
    summary="Clear chat history for a collection or session",
)
async def clear_chat_history(
    collection_id: UUID,
    session_id: Optional[str] = Query(None, description="Optional session filter"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    col_str = str(collection_id)
    stmt = delete(ChatMessageRecord).where(ChatMessageRecord.collection_id == col_str)
    if session_id:
        stmt = stmt.where(ChatMessageRecord.session_id == session_id)
    result = await db.execute(stmt)
    await db.commit()
    return {"status": "cleared", "deleted_count": result.rowcount}
