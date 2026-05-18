"""routers/chat.py"""
from __future__ import annotations
import json, logging, uuid
from fastapi import APIRouter, Depends, HTTPException
from core.auth import get_current_user
from core.database import DB
from models.schemas import MessageRequest, MessageResponse, SessionCreate, SessionListOut, MessageListOut
from services.rag import answer_question

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sessions")
async def create_session(body: SessionCreate, user=Depends(get_current_user)):
    # RLS: "Users can manage their own chat sessions"
    session = DB.create_session({"user_id":user["id"],"title":body.title}, user["jwt"])
    return session


@router.get("/sessions", response_model=SessionListOut)
async def list_sessions(user=Depends(get_current_user)):
    sessions = DB.list_sessions(user["id"], user["jwt"])
    return SessionListOut(sessions=sessions, total=len(sessions))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user=Depends(get_current_user)):
    sess = DB.get_session(session_id, user["jwt"])
    if not sess: raise HTTPException(404, "Session not found")
    DB.delete_session(session_id, user["jwt"])   # CASCADE removes messages
    return {"message":"Session deleted","session_id":session_id}


@router.get("/sessions/{session_id}/messages", response_model=MessageListOut)
async def list_messages(session_id: str, user=Depends(get_current_user)):
    # RLS: EXISTS (chat_sessions WHERE user_id = auth.uid())
    sess = DB.get_session(session_id, user["jwt"])
    if not sess: raise HTTPException(404,"Session not found")
    msgs = DB.list_messages(session_id, user["jwt"])
    return MessageListOut(messages=msgs, total=len(msgs))


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: str, body: MessageRequest, user=Depends(get_current_user)):
    """
    MAIN CHAT ENDPOINT
    ──────────────────
    1. Verify session belongs to user (RLS enforced inside DB.get_session).
    2. Persist user message.
    3. Run RAG pipeline (embed → Pinecone → GPT-4o).
    4. Persist assistant message with citations JSONB.
    5. Touch session updated_at.
    6. Return MessageResponse to Flutter.

    All DB reads/writes use the user's JWT → Supabase RLS enforced:
      messages → EXISTS (chat_sessions WHERE user_id = auth.uid())
    """
    jwt = user["jwt"]

    # ── Verify session ────────────────────────────────────────────────────────
    sess = DB.get_session(session_id, jwt)
    if not sess: raise HTTPException(404,"Session not found")

    # ── Fetch user settings for language / detail_level ───────────────────────
    user_row = DB.get_user(user["id"], jwt) or {}
    user_settings = user_row.get("settings", {"language":"ar","detail_level":"detailed"})

    # ── Persist user message ──────────────────────────────────────────────────
    user_msg = DB.insert_message(
        {"session_id":session_id,"role":"user","content":body.question,"citations":[]}, jwt
    )

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    answer, citations, confidence, tokens = await answer_question(
        question=body.question,
        session_id=session_id,
        user_id=user["id"],
        jwt=jwt,
        user_settings=user_settings,
    )

    # ── Persist assistant message (service client — pipeline context) ─────────
    ai_msg = DB.insert_message_service({
        "session_id": session_id,
        "role":       "assistant",
        "content":    answer,
        "citations":  json.dumps([c.model_dump() for c in citations]),
    })

    DB.touch_session(session_id)

    return MessageResponse(
        session_id=session_id,
        message_id=str(ai_msg["id"]),
        answer=answer,
        citations=citations,
        confidence_score=confidence,
        tokens_used=tokens,
    )
