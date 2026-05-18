"""
models/schemas.py
Pydantic v2 models — mirror the exact Supabase SQL schema.
Used for request validation and response serialisation throughout the API.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────────────────────
class OrmBase(BaseModel):
    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=2, max_length=100)
    specialty: str = "general"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserOut"


# ─────────────────────────────────────────────────────────────────────────────
# USERS  →  public.users
# ─────────────────────────────────────────────────────────────────────────────
class UserSettings(BaseModel):
    language: str = "ar"          # ar | en
    detail_level: str = "detailed"  # concise | balanced | detailed


class UserOut(OrmBase):
    id: UUID
    email: EmailStr
    full_name: Optional[str] = None
    specialty: str = "general"
    settings: UserSettings = UserSettings()
    created_at: datetime


class UserSettingsUpdate(BaseModel):
    language: Optional[str] = None
    detail_level: Optional[str] = None
    specialty: Optional[str] = None
    full_name: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTS  →  public.documents
# ─────────────────────────────────────────────────────────────────────────────
class DocumentStatus:
    PENDING    = "pending"
    EXTRACTING = "extracting"
    INDEXING   = "indexing"
    READY      = "ready"
    FAILED     = "failed"


class DocumentOut(OrmBase):
    id: UUID
    user_id: UUID
    name: str
    type: Optional[str] = None
    storage_path: str
    status: str
    file_size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    uploaded_at: datetime
    indexed_at: Optional[datetime] = None


class DocumentListOut(BaseModel):
    documents: List[DocumentOut]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKS  →  public.chunks
# ─────────────────────────────────────────────────────────────────────────────
class ChunkOut(OrmBase):
    id: UUID
    document_id: UUID
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    pinecone_id: str
    token_count: Optional[int] = None
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# CITATIONS  (embedded inside messages.citations JSONB)
# ─────────────────────────────────────────────────────────────────────────────
class Citation(BaseModel):
    """One retrieved source chunk returned alongside an AI answer."""
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    pinecone_id: str


# ─────────────────────────────────────────────────────────────────────────────
# CHAT SESSIONS  →  public.chat_sessions
# ─────────────────────────────────────────────────────────────────────────────
class SessionOut(OrmBase):
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class SessionCreate(BaseModel):
    title: str = "محادثة جديدة"


class SessionListOut(BaseModel):
    sessions: List[SessionOut]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGES  →  public.messages
# ─────────────────────────────────────────────────────────────────────────────
class MessageOut(OrmBase):
    id: UUID
    session_id: UUID
    role: str                          # user | assistant
    content: str
    citations: List[Citation] = []
    created_at: datetime


class MessageRequest(BaseModel):
    """Body sent by the Flutter app when the user submits a question."""
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None   # omit to auto-create session


class MessageResponse(BaseModel):
    """Full response returned to the Flutter app."""
    session_id: str
    message_id: str
    answer: str
    citations: List[Citation]
    confidence_score: float            # 0–1 composite score
    tokens_used: int


class MessageListOut(BaseModel):
    messages: List[MessageOut]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK  →  public.feedback
# ─────────────────────────────────────────────────────────────────────────────
class FeedbackType:
    POSITIVE = "positive"
    NEGATIVE = "negative"


class FeedbackRequest(BaseModel):
    message_id: str
    type: str                          # positive | negative
    reason: Optional[str] = None       # only for negative
    comment: Optional[str] = None
    # snapshots for internal analysis
    question: Optional[str] = None
    answer: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in (FeedbackType.POSITIVE, FeedbackType.NEGATIVE):
            raise ValueError("type must be 'positive' or 'negative'")
        return v


class FeedbackOut(OrmBase):
    id: UUID
    message_id: UUID
    user_id: UUID
    type: str
    reason: Optional[str] = None
    comment: Optional[str] = None
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD progress event (for Supabase Realtime / SSE)
# ─────────────────────────────────────────────────────────────────────────────
class ProcessingEvent(BaseModel):
    document_id: str
    status: str
    progress_pct: int = 0              # 0–100
    message: str = ""
    error: Optional[str] = None
