"""
backend/schemas/chat_history.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for Chat History and Conversation Logging (Phase 17).
─────────────────────────────────────────────────────────────
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatMessageCreate(BaseModel):
    """Payload to create a chat history message."""
    session_id: str = Field(..., description="Unique conversation session UUID string.")
    role: str = Field(..., pattern="^(user|assistant)$", description="'user' or 'assistant'.")
    content: str = Field(..., min_length=1, description="Message text.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata (sources, citations, trace).")

    model_config = ConfigDict(from_attributes=True)


class ChatMessageResponse(BaseModel):
    """Structured representation of a stored chat message."""
    id: str
    collection_id: str
    session_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatHistoryResponse(BaseModel):
    """List of messages for an active collection or session."""
    collection_id: str
    session_id: Optional[str] = None
    total: int
    messages: List[ChatMessageResponse]

    model_config = ConfigDict(from_attributes=True)
