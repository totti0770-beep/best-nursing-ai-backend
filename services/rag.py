"""
services/rag.py  —  RAG query pipeline
Steps: embed → search → hydrate → history → GPT-4o → citations
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from pinecone import Pinecone

from core.config import settings
from core.database import DB
from models.schemas import Citation

logger = logging.getLogger(__name__)

_openai: OpenAI | None = None
_pinecone_index = None

def _oai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai

def _idx():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _pinecone_index

# ── System prompt ─────────────────────────────────────────────────────────────
_SYS = """\
You are a specialized clinical AI assistant for nurses and healthcare professionals.

RULES:
- Only answer using the CONTEXT CHUNKS provided below.
- Cite every fact: [Doc: <name>, Page: <n>]
- Respond in {language} with {detail} level of detail.
- Use bold headings (** **) and bullet points.
- Add ⚠️ warning boxes for high-risk interventions.
- If context is insufficient say: "هذه المعلومات غير متوفرة في مستنداتك المرفوعة."
- Never invent dosages or contraindications.
"""
_DISCLAIMER = "\n\n---\n⚕️ *للأغراض المرجعية فقط — لا تُغني عن الحكم السريري.*"
_DETAIL_MAP = {"concise":"concise (2-3 sentences)","balanced":"balanced","detailed":"comprehensive"}
_LANG_MAP   = {"ar":"Arabic","en":"English"}

def _sys(lang: str, detail: str) -> str:
    return _SYS.format(
        language=_LANG_MAP.get(lang,"Arabic"),
        detail=_DETAIL_MAP.get(detail,"balanced"),
    )

async def answer_question(
    question: str,
    session_id: str,
    user_id: str,
    jwt: str,
    user_settings: Optional[Dict] = None,
) -> Tuple[str, List[Citation], float, int]:
    """Returns (answer, citations, confidence_score, tokens_used)."""

    s = user_settings or {}
    lang   = s.get("language","ar")
    detail = s.get("detail_level","detailed")

    # 1 — embed query
    q_vec = _oai().embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=question,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    ).data[0].embedding

    # 2 — vector search (namespace = user_id → user isolation)
    result = _idx().query(
        vector=q_vec, top_k=settings.TOP_K_CHUNKS,
        namespace=user_id, include_metadata=True,
    )
    matches = [m for m in result.matches if m.score >= settings.MIN_RELEVANCE_SCORE]

    if not matches:
        no_ans = "لم أجد معلومات كافية في مستنداتك للإجابة على هذا السؤال." + _DISCLAIMER
        return no_ans, [], 0.0, 0

    # 3+4 — hydrate chunks + document names
    pids   = [m.id for m in matches]
    scores = {m.id: m.score for m in matches}
    chunks = DB.get_chunks_by_pinecone_ids(pids)
    doc_ids = list({c["document_id"] for c in chunks})
    doc_rows = (
        DB.get_service_client()
        .table("documents").select("id,name")
        .in_("id", doc_ids).execute().data
    )
    dmap = {d["id"]: d["name"] for d in doc_rows}
    hydrated = [
        {**c,
         "document_name": dmap.get(c["document_id"],"Unknown"),
         "score": scores.get(c["pinecone_id"],0.0)}
        for c in chunks
    ]

    # 5 — conversation memory (last 6 turns)
    history = [
        {"role": r["role"], "content": r["content"]}
        for r in DB.list_messages(session_id, jwt, limit=6)
    ]

    # 6 — build prompt
    ctx_block = "\n\n".join(
        f"[{i+1}] {c['document_name']} | Page {c['page_number']}\n{c['text']}"
        for i, c in enumerate(hydrated)
    )
    user_msg = f"CONTEXT CHUNKS:\n{ctx_block}\n\n---\nQUESTION: {question}"

    # 7 — GPT-4o
    resp = _oai().chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[
            {"role":"system","content":_sys(lang,detail)},
            *history,
            {"role":"user","content":user_msg},
        ],
        temperature=settings.CHAT_TEMPERATURE,
        max_tokens=settings.CHAT_MAX_TOKENS,
    )
    answer  = resp.choices[0].message.content + _DISCLAIMER
    tokens  = resp.usage.total_tokens

    # 8 — citations
    cits = [
        Citation(
            document_id=c["document_id"],
            document_name=c["document_name"],
            page_number=c["page_number"],
            excerpt=c["text"][:300],
            relevance_score=round(c["score"],4),
            pinecone_id=c["pinecone_id"],
        )
        for c in hydrated
    ]
    conf = round(sum(c["score"] for c in hydrated) / len(hydrated), 4)
    logger.info(f"[RAG] tokens={tokens} cits={len(cits)} conf={conf}")
    return answer, cits, conf, tokens
