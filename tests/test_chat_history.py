"""
tests/test_chat_history.py
─────────────────────────────────────────────────────────────
Unit and integration tests for Phase 17: Chat History and Session Persistence.
─────────────────────────────────────────────────────────────
"""

from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient

from backend.database import init_db
from backend.main import app
from backend.models.collection import Collection


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()


@pytest.mark.asyncio
async def test_chat_history_nonexistent_collection_returns_404():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        fake_id = str(uuid4())
        resp = await client.get(f"/api/collections/{fake_id}/chat/history")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_chat_history_save_and_retrieve_flow():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create a test collection
        col_name = f"Test Collection {uuid4().hex[:6]}"
        create_resp = await client.post(
            "/api/collections",
            json={"name": col_name, "description": "Chat history test collection"},
        )
        assert create_resp.status_code == 201
        col_id = create_resp.json()["id"]

        session_id = f"session-{uuid4().hex[:8]}"

        # Save user message
        msg1_resp = await client.post(
            f"/api/collections/{col_id}/chat/history",
            json={
                "session_id": session_id,
                "role": "user",
                "content": "What was the operating revenue in Q3?",
            },
        )
        assert msg1_resp.status_code == 201
        msg1_data = msg1_resp.json()
        assert msg1_data["role"] == "user"
        assert msg1_data["session_id"] == session_id

        # Save assistant message with metadata
        meta = {
            "valid_citations": ["[S1]"],
            "grounding_score": 0.95,
            "sources": [{"source_id": "[S1]", "char_count": 80}],
        }
        msg2_resp = await client.post(
            f"/api/collections/{col_id}/chat/history",
            json={
                "session_id": session_id,
                "role": "assistant",
                "content": "Operating revenue was $4.2B [S1].",
                "metadata": meta,
            },
        )
        assert msg2_resp.status_code == 201
        msg2_data = msg2_resp.json()
        assert msg2_data["role"] == "assistant"
        assert msg2_data["metadata"]["grounding_score"] == 0.95

        # Retrieve chat history
        hist_resp = await client.get(f"/api/collections/{col_id}/chat/history?session_id={session_id}")
        assert hist_resp.status_code == 200
        hist_data = hist_resp.json()
        assert hist_data["total"] == 2
        assert len(hist_data["messages"]) == 2
        assert hist_data["messages"][0]["content"] == "What was the operating revenue in Q3?"
        assert hist_data["messages"][1]["content"] == "Operating revenue was $4.2B [S1]."

        # Clear chat history
        del_resp = await client.delete(f"/api/collections/{col_id}/chat/history?session_id={session_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "cleared"

        # Verify history is now empty
        empty_resp = await client.get(f"/api/collections/{col_id}/chat/history?session_id={session_id}")
        assert empty_resp.status_code == 200
        assert empty_resp.json()["total"] == 0
