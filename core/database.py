"""
core/database.py
─────────────────────────────────────────────────────────────────────────────
Two Supabase clients:
  anon_client    → ANON key   → RLS enforced  → auth flows
  service_client → SERVICE key → bypasses RLS → background tasks / admin

Every user-facing method receives the user JWT and calls get_user_client(jwt)
so that ALL six RLS policies defined in the schema are enforced automatically:

  users         → auth.uid() = id
  documents     → auth.uid() = user_id
  chunks        → EXISTS (documents WHERE user_id = auth.uid())   [SELECT only]
  chat_sessions → auth.uid() = user_id
  messages      → EXISTS (chat_sessions WHERE user_id = auth.uid())
  feedback      → auth.uid() = user_id  (to be added)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from supabase import create_client, Client
from core.config import settings

logger = logging.getLogger(__name__)

_anon_client: Client | None = None
_service_client: Client | None = None


def get_anon_client() -> Client:
    global _anon_client
    if _anon_client is None:
        _anon_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    return _anon_client


def get_service_client() -> Client:
    """SERVICE_ROLE key — bypasses RLS. Only use in background tasks."""
    global _service_client
    if _service_client is None:
        _service_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    return _service_client


def get_user_client(jwt: str) -> Client:
    """
    Anon client with the user's JWT injected.
    Supabase will call auth.uid() = <jwt sub> for every RLS check.
    """
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    client.auth.set_session(jwt, "")
    return client


async def init_db() -> None:
    try:
        svc = get_service_client()
        r = svc.table("users").select("id", count="exact").limit(1).execute()
        logger.info(f"Supabase connected — users table rows: {r.count}")
    except Exception as exc:
        logger.error(f"Supabase connection failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
class DB:
    """
    Thin wrapper. Every method documents which RLS policy gates the query.
    """

    # ── users ─────────────────────────────────────────────────────────────────
    @staticmethod
    def get_user(user_id: str, jwt: str) -> Optional[Dict]:
        # RLS: "Users can manage their own profile" → auth.uid() = id
        r = (get_user_client(jwt).table("users")
             .select("*").eq("id", user_id).single().execute())
        return r.data

    @staticmethod
    def create_user(payload: Dict) -> Dict:
        # Called after Supabase Auth signup — no JWT yet → service client
        r = get_service_client().table("users").insert(payload).execute()
        return r.data[0]

    @staticmethod
    def update_user(user_id: str, payload: Dict, jwt: str) -> Dict:
        # RLS: auth.uid() = id
        r = (get_user_client(jwt).table("users")
             .update(payload).eq("id", user_id).execute())
        return r.data[0]

    # ── documents ─────────────────────────────────────────────────────────────
    @staticmethod
    def create_document(payload: Dict, jwt: str) -> Dict:
        # RLS: "Users can manage their own documents" → auth.uid() = user_id
        r = (get_user_client(jwt).table("documents").insert(payload).execute())
        return r.data[0]

    @staticmethod
    def list_documents(user_id: str, jwt: str) -> List[Dict]:
        # RLS: auth.uid() = user_id  (WHERE clause redundant but explicit)
        r = (get_user_client(jwt).table("documents")
             .select("*").eq("user_id", user_id)
             .order("uploaded_at", desc=True).execute())
        return r.data

    @staticmethod
    def get_document(doc_id: str, jwt: str) -> Optional[Dict]:
        # RLS: auth.uid() = user_id
        r = (get_user_client(jwt).table("documents")
             .select("*").eq("id", doc_id).single().execute())
        return r.data

    @staticmethod
    def update_document_status(
        doc_id: str,
        status: str,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Background processing tasks call this. Uses SERVICE client because
        the pipeline runs outside a request/JWT context.
        RLS bypassed intentionally — access was verified at upload time.
        """
        payload: Dict[str, Any] = {"status": status}
        if extra:
            payload.update(extra)
        get_service_client().table("documents").update(payload).eq("id", doc_id).execute()
        logger.info(f"Document {doc_id} status → {status}")

    @staticmethod
    def delete_document(doc_id: str, jwt: str) -> None:
        # RLS: auth.uid() = user_id  |  CASCADE deletes chunks
        get_user_client(jwt).table("documents").delete().eq("id", doc_id).execute()

    # ── chunks ────────────────────────────────────────────────────────────────
    @staticmethod
    def insert_chunks(chunks: List[Dict]) -> None:
        """
        Bulk INSERT from the processing pipeline.
        SERVICE client — RLS on chunks is SELECT-only for users:
          EXISTS (SELECT 1 FROM documents
                  WHERE documents.id = chunks.document_id
                  AND documents.user_id = auth.uid())
        Writes always come from service role, never directly from the user.
        """
        get_service_client().table("chunks").insert(chunks).execute()

    @staticmethod
    def get_chunks_by_pinecone_ids(pinecone_ids: List[str]) -> List[Dict]:
        """RAG pipeline — fetches chunk text after vector search. Service client."""
        r = (get_service_client().table("chunks")
             .select("id, document_id, text, page_number, pinecone_id, token_count")
             .in_("pinecone_id", pinecone_ids).execute())
        return r.data

    @staticmethod
    def delete_chunks_by_document(doc_id: str) -> None:
        """Called before re-indexing. Service client."""
        get_service_client().table("chunks").delete().eq("document_id", doc_id).execute()

    # ── chat_sessions ─────────────────────────────────────────────────────────
    @staticmethod
    def create_session(payload: Dict, jwt: str) -> Dict:
        # RLS: "Users can manage their own chat sessions" → auth.uid() = user_id
        r = (get_user_client(jwt).table("chat_sessions").insert(payload).execute())
        return r.data[0]

    @staticmethod
    def list_sessions(user_id: str, jwt: str) -> List[Dict]:
        # RLS: auth.uid() = user_id
        r = (get_user_client(jwt).table("chat_sessions")
             .select("*").eq("user_id", user_id)
             .order("updated_at", desc=True).execute())
        return r.data

    @staticmethod
    def get_session(session_id: str, jwt: str) -> Optional[Dict]:
        # RLS: auth.uid() = user_id
        r = (get_user_client(jwt).table("chat_sessions")
             .select("*").eq("id", session_id).single().execute())
        return r.data

    @staticmethod
    def delete_session(session_id: str, jwt: str) -> None:
        # RLS: auth.uid() = user_id  |  CASCADE deletes messages
        get_user_client(jwt).table("chat_sessions").delete().eq("id", session_id).execute()

    @staticmethod
    def touch_session(session_id: str) -> None:
        """Update updated_at. Service client — called from RAG pipeline."""
        get_service_client().table("chat_sessions").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", session_id).execute()

    # ── messages ──────────────────────────────────────────────────────────────
    @staticmethod
    def insert_message(payload: Dict, jwt: str) -> Dict:
        # RLS: "Users can manage messages in their sessions"
        #  EXISTS (chat_sessions WHERE user_id = auth.uid())
        r = (get_user_client(jwt).table("messages").insert(payload).execute())
        return r.data[0]

    @staticmethod
    def insert_message_service(payload: Dict) -> Dict:
        """RAG pipeline inserts assistant reply. Session ownership already verified."""
        r = get_service_client().table("messages").insert(payload).execute()
        return r.data[0]

    @staticmethod
    def list_messages(session_id: str, jwt: str, limit: int = 20) -> List[Dict]:
        # RLS: EXISTS (chat_sessions WHERE user_id = auth.uid())
        r = (get_user_client(jwt).table("messages")
             .select("*").eq("session_id", session_id)
             .order("created_at", desc=False).limit(limit).execute())
        return r.data

    # ── feedback ──────────────────────────────────────────────────────────────
    @staticmethod
    def insert_feedback(payload: Dict, jwt: str) -> Dict:
        # RLS: auth.uid() = user_id  (add policy if not yet created)
        r = (get_user_client(jwt).table("feedback").insert(payload).execute())
        return r.data[0]

    @staticmethod
    def list_feedback(limit: int = 200) -> List[Dict]:
        """Admin endpoint — service client bypasses RLS."""
        r = (get_service_client().table("feedback")
             .select("*").order("created_at", desc=True).limit(limit).execute())
        return r.data
