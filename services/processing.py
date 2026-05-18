"""
services/processing.py
─────────────────────────────────────────────────────────────────────────────
Document ingestion pipeline (runs as a FastAPI BackgroundTask):

  1. Update status → "extracting"
  2. Extract text page-by-page with pdfplumber
  3. Clean and normalise text
  4. Chunk with LangChain RecursiveCharacterTextSplitter
  5. Embed chunks with OpenAI text-embedding-3-large
  6. Upsert vectors to Pinecone (namespace = user_id)
  7. Store chunk rows in public.chunks
  8. Update document → status="ready", chunk_count, page_count, indexed_at
  9. On any error → status="failed"

Supabase Realtime listeners in Flutter will pick up each status change
and update the UI automatically.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from core.config import settings
from core.database import DB

logger = logging.getLogger(__name__)

# ── Singleton clients (created once per worker process) ───────────────────────
_openai: OpenAI | None = None
_pinecone_index = None


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSIONS,   # 3072
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Created Pinecone index: {settings.PINECONE_INDEX_NAME}")
        _pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _pinecone_index


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> Tuple[str, int]:
    """
    Returns (full_text, page_count).
    Preserves page boundaries with a sentinel so chunker can track page numbers.
    """
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages.append(raw)

    full_text = "\n\n<<PAGE_BREAK>>\n\n".join(pages)
    return full_text, len(pages)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Text cleaning
# ─────────────────────────────────────────────────────────────────────────────
_HEADER_FOOTER = re.compile(
    r"^(Page\s+\d+|©.*|Confidential.*|www\.\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_text(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = _HEADER_FOOTER.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Chunking
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(full_text: str) -> List[Dict]:
    """
    Returns list of:
      { "text": str, "page_number": int, "chunk_index": int }

    Strategy: split on <<PAGE_BREAK>> first (to track page numbers), then
    apply LangChain token-aware splitter within each page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,          # 512 tokens
        chunk_overlap=settings.CHUNK_OVERLAP,    # 50 tokens
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pages = full_text.split("<<PAGE_BREAK>>")
    results: List[Dict] = []
    global_idx = 0

    for page_num, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        sub_chunks = splitter.split_text(page_text)
        for chunk in sub_chunks:
            chunk = chunk.strip()
            if len(chunk) < 30:   # skip meaningless fragments
                continue
            results.append({
                "text": chunk,
                "page_number": page_num,
                "chunk_index": global_idx,
            })
            global_idx += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Embedding
# ─────────────────────────────────────────────────────────────────────────────
def embed_chunks(texts: List[str]) -> List[List[float]]:
    """
    Calls OpenAI Embeddings API in batches of 100.
    Model: text-embedding-3-large (3072 dimensions).
    """
    all_vectors: List[List[float]] = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _get_openai().embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        all_vectors.extend([e.embedding for e in resp.data])

    return all_vectors


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Pinecone upsert
# ─────────────────────────────────────────────────────────────────────────────
def upsert_vectors(
    chunks: List[Dict],
    vectors: List[List[float]],
    document_id: str,
    user_id: str,
) -> List[str]:
    """
    Upsert chunk vectors to Pinecone.
    Namespace = user_id  → enforces logical data isolation per user.

    Returns list of pinecone_ids (one per chunk).
    """
    index = _get_pinecone_index()
    pinecone_ids: List[str] = []
    upsert_data = []

    for chunk, vector in zip(chunks, vectors):
        # Deterministic ID: hash(doc_id + chunk_index)
        pid = hashlib.sha256(
            f"{document_id}:{chunk['chunk_index']}".encode()
        ).hexdigest()[:40]
        pinecone_ids.append(pid)

        upsert_data.append({
            "id": pid,
            "values": vector,
            "metadata": {
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "text_preview": chunk["text"][:200],  # for debugging
            },
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(upsert_data), batch_size):
        index.upsert(
            vectors=upsert_data[i : i + batch_size],
            namespace=user_id,
        )

    return pinecone_ids


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Persist chunks to Supabase
# ─────────────────────────────────────────────────────────────────────────────
def persist_chunks(
    chunks: List[Dict],
    pinecone_ids: List[str],
    document_id: str,
) -> None:
    rows = [
        {
            "document_id": document_id,
            "chunk_index": ch["chunk_index"],
            "text": ch["text"],
            "page_number": ch["page_number"],
            "pinecone_id": pid,
            "token_count": len(ch["text"].split()),  # rough token estimate
        }
        for ch, pid in zip(chunks, pinecone_ids)
    ]
    DB.insert_chunks(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry-point  (called by BackgroundTasks)
# ─────────────────────────────────────────────────────────────────────────────
async def process_document(
    doc_id: str,
    tmp_path: str,
    user_id: str,
) -> None:
    """
    Full ingestion pipeline.  All DB writes use the service client.
    Status transitions broadcast via Supabase Realtime → Flutter app updates UI.

    Status flow:
      pending → extracting → indexing → ready
                                      → failed (on any exception)
    """
    logger.info(f"[Pipeline] Starting  doc={doc_id}  user={user_id}")

    try:
        # ── 1. Extracting ─────────────────────────────────────────────────────
        DB.update_document_status(doc_id, "extracting")
        logger.info(f"[Pipeline] Extracting text …")

        full_text, page_count = extract_text(tmp_path)
        full_text = clean_text(full_text)

        if not full_text.strip():
            raise ValueError("PDF contains no extractable text (possibly scanned image)")

        # ── 2. Chunking ───────────────────────────────────────────────────────
        DB.update_document_status(doc_id, "indexing")
        logger.info(f"[Pipeline] Chunking ({page_count} pages) …")

        chunks = chunk_text(full_text)
        logger.info(f"[Pipeline] {len(chunks)} chunks produced")

        # ── 3. Embedding ──────────────────────────────────────────────────────
        logger.info(f"[Pipeline] Embedding {len(chunks)} chunks …")
        texts = [c["text"] for c in chunks]
        vectors = embed_chunks(texts)

        # ── 4. Pinecone upsert ────────────────────────────────────────────────
        logger.info("[Pipeline] Upserting to Pinecone …")
        pinecone_ids = upsert_vectors(chunks, vectors, doc_id, user_id)

        # ── 5. Persist chunks ─────────────────────────────────────────────────
        logger.info("[Pipeline] Persisting chunks to Supabase …")
        persist_chunks(chunks, pinecone_ids, doc_id)

        # ── 6. Mark ready ─────────────────────────────────────────────────────
        DB.update_document_status(
            doc_id,
            "ready",
            extra={
                "page_count": page_count,
                "chunk_count": len(chunks),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(f"[Pipeline] ✅ Done  doc={doc_id}  chunks={len(chunks)}")

    except Exception as exc:
        logger.error(f"[Pipeline] ❌ Failed  doc={doc_id}: {exc}", exc_info=True)
        DB.update_document_status(doc_id, "failed")

    finally:
        # Always clean up the temporary file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
