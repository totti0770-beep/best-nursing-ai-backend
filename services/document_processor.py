"""
services/document_processor.py
═══════════════════════════════
Complete RAG ingestion pipeline executed as a FastAPI BackgroundTask.

Flow:
  PDF upload → extract text → chunk → embed → upsert Pinecone → store chunks in Supabase

Status transitions written to public.documents:
  pending → extracting → indexing → ready  (or → failed)
"""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Generator, List, Tuple

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec

from core.config import settings
from core.database import get_supabase

logger = logging.getLogger(__name__)

# ── Clients (initialised lazily so tests can mock them) ───────────────────────
_openai: AsyncOpenAI | None = None
_pinecone_index = None


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        # Create index if it doesn't exist
        if settings.PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSIONS,  # 3072 for text-embedding-3-large
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _pinecone_index


# ─────────────────────────────────────────────────────────────────────────────
# Status helpers — write to public.documents
# ─────────────────────────────────────────────────────────────────────────────
def _set_status(doc_id: str, status: str, extra: dict | None = None) -> None:
    """Service-role client — bypasses RLS intentionally for background jobs."""
    sb = get_supabase()
    payload = {"status": status, **(extra or {})}
    sb.table("documents").update(payload).eq("id", doc_id).execute()
    logger.info(f"[doc:{doc_id}] status → {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract text from PDF
# ─────────────────────────────────────────────────────────────────────────────
def _extract_text(pdf_path: str) -> Tuple[List[Tuple[int, str]], int]:
    """
    Returns:
        pages   : list of (page_number, cleaned_text)
        n_pages : total page count
    """
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            if cleaned.strip():
                pages.append((i, cleaned))
    return pages, n_pages


def _clean_text(text: str) -> str:
    """Remove headers/footers noise and normalise whitespace."""
    # Drop lines that look like page numbers: "12", "Page 12", "- 12 -"
    text = re.sub(r"(?m)^[\s\-]*Page\s*\d+[\s\-]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix broken hyphenation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Chunk text with LangChain RecursiveCharacterTextSplitter
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_pages(
    pages: List[Tuple[int, str]],
) -> List[dict]:
    """
    Returns a list of chunk dicts:
      { text, page_number, chunk_index }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,              # character-based; swap for tiktoken if preferred
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[dict] = []
    idx = 0
    for page_num, page_text in pages:
        for fragment in splitter.split_text(page_text):
            if fragment.strip():
                chunks.append(
                    {
                        "text": fragment.strip(),
                        "page_number": page_num,
                        "chunk_index": idx,
                    }
                )
                idx += 1
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Embed with OpenAI text-embedding-3-large
# ─────────────────────────────────────────────────────────────────────────────
async def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts. OpenAI allows up to 2048 texts per call,
    but we batch conservatively at 100 to stay within rate limits.
    """
    client = _get_openai()
    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def _batch(lst: list, size: int) -> Generator:
    """Yield successive fixed-size slices from a list."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Upsert vectors to Pinecone
# ─────────────────────────────────────────────────────────────────────────────
def _upsert_to_pinecone(
    vectors: List[Tuple[str, List[float], dict]],
    namespace: str,
) -> None:
    """
    vectors : list of (pinecone_id, embedding, metadata_dict)
    namespace: user_id string → isolates each user's vectors
    """
    index = _get_pinecone_index()
    # Pinecone upsert_batch size recommendation: ≤ 100 per call
    for batch in _batch(vectors, 100):
        records = [
            {"id": pid, "values": emb, "metadata": meta}
            for pid, emb, meta in batch
        ]
        index.upsert(vectors=records, namespace=namespace)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Persist chunks to public.chunks
# ─────────────────────────────────────────────────────────────────────────────
def _save_chunks(doc_id: str, chunk_records: List[dict]) -> None:
    """
    chunk_records shape:
      { document_id, chunk_index, text, page_number, pinecone_id, token_count }
    Uses service-role client — RLS policy for chunks is SELECT-only for users;
    the backend inserts with elevated privileges.
    """
    sb = get_supabase()
    # Insert in batches of 200 to avoid request-size limits
    for batch in _batch(chunk_records, 200):
        sb.table("chunks").insert(batch).execute()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline — called by BackgroundTasks
# ─────────────────────────────────────────────────────────────────────────────
async def process_document(doc_id: str, pdf_path: str, user_id: str) -> None:
    """
    Full ingestion pipeline. Catches ALL exceptions so the background task
    never crashes the worker; on failure it writes status='failed' to the DB.

    Called from routers/documents.py:
        background_tasks.add_task(process_document, doc_id, tmp_path, user_id)
    """
    try:
        # ── Stage 1: Extract ──────────────────────────────────────────────────
        _set_status(doc_id, "extracting")
        pages, n_pages = _extract_text(pdf_path)
        if not pages:
            raise ValueError("PDF appears to be empty or contains only images.")

        # ── Stage 2: Chunk ────────────────────────────────────────────────────
        raw_chunks = _chunk_pages(pages)
        logger.info(f"[doc:{doc_id}] {len(raw_chunks)} chunks from {n_pages} pages")

        # ── Stage 3: Embed + Upsert Pinecone (batched) ───────────────────────
        _set_status(doc_id, "indexing")
        pinecone_vectors: List[Tuple[str, List[float], dict]] = []
        chunk_records: List[dict] = []

        EMBED_BATCH = 100
        for batch_chunks in _batch(raw_chunks, EMBED_BATCH):
            texts = [c["text"] for c in batch_chunks]
            embeddings = await _embed_batch(texts)

            for chunk, emb in zip(batch_chunks, embeddings):
                pinecone_id = f"{doc_id}_{chunk['chunk_index']}"
                token_count = math.ceil(len(chunk["text"]) / 4)  # rough estimate

                # Pinecone vector + metadata
                pinecone_vectors.append((
                    pinecone_id,
                    emb,
                    {
                        "document_id": doc_id,
                        "user_id": user_id,
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"][:500],   # Pinecone metadata cap
                    },
                ))

                # Supabase chunk row
                chunk_records.append({
                    "document_id": doc_id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "pinecone_id": pinecone_id,
                    "token_count": token_count,
                })

        # ── Stage 4: Upsert Pinecone ──────────────────────────────────────────
        _upsert_to_pinecone(pinecone_vectors, namespace=user_id)

        # ── Stage 5: Persist chunks to Supabase ──────────────────────────────
        _save_chunks(doc_id, chunk_records)

        # ── Done ──────────────────────────────────────────────────────────────
        from datetime import datetime, timezone
        _set_status(
            doc_id,
            "ready",
            {
                "page_count": n_pages,
                "chunk_count": len(chunk_records),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(f"[doc:{doc_id}] ✅ processing complete — {len(chunk_records)} chunks indexed")

    except Exception as exc:
        logger.error(f"[doc:{doc_id}] ❌ processing failed: {exc}", exc_info=True)
        _set_status(doc_id, "failed", {"status": "failed"})
    finally:
        # Always clean up the temp file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except OSError:
            pass
