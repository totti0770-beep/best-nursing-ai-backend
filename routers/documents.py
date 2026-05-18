"""routers/documents.py"""
from __future__ import annotations
import logging, os, uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from core.auth import get_current_user
from core.config import settings
from core.database import DB, get_service_client
from models.schemas import DocumentListOut, DocumentOut
from services.processing import process_document

router   = APIRouter()
logger   = logging.getLogger(__name__)
MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024

def _del_pinecone(user_id: str, doc_id: str):
    try:
        from pinecone import Pinecone
        idx = Pinecone(api_key=settings.PINECONE_API_KEY).Index(settings.PINECONE_INDEX_NAME)
        rows = get_service_client().table("chunks").select("pinecone_id").eq("document_id",doc_id).execute().data
        pids = [r["pinecone_id"] for r in rows]
        if pids: idx.delete(ids=pids, namespace=user_id)
    except Exception as e: logger.warning(f"Pinecone cleanup: {e}")


@router.post("/upload", status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form(default="guideline"),
    user=Depends(get_current_user),
):
    """
    UPLOAD FLOW:
      validate → save tmp → Supabase Storage → insert documents row (status=pending)
      → BackgroundTask(process_document) → return 201 immediately

    Flutter subscribes to Supabase Realtime on public.documents and watches
    status:  pending → extracting → indexing → ready  (or failed)
    """
    if file.content_type not in {"application/pdf"}:
        raise HTTPException(415, f"Only PDFs accepted. Got: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {settings.MAX_FILE_SIZE_MB} MB")

    doc_id  = str(uuid.uuid4())
    user_id = user["id"]; jwt = user["jwt"]

    # ── tmp file ──────────────────────────────────────────────────────────────
    os.makedirs(settings.TMP_DIR, exist_ok=True)
    tmp_path = f"{settings.TMP_DIR}/{doc_id}.pdf"
    with open(tmp_path,"wb") as fh: fh.write(content)

    # ── Supabase Storage ──────────────────────────────────────────────────────
    storage_path = f"{user_id}/{doc_id}/{file.filename}"
    try:
        get_service_client().storage.from_(settings.SUPABASE_BUCKET).upload(
            storage_path, content, {"content-type":"application/pdf"})
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(500, f"Storage upload failed: {e}")

    # ── INSERT documents row (RLS: auth.uid() = user_id) ─────────────────────
    DB.create_document({"id":doc_id,"user_id":user_id,"name":file.filename,
        "type":doc_type,"storage_path":storage_path,"status":"pending",
        "file_size_bytes":len(content)}, jwt)

    # ── BackgroundTask → pipeline ─────────────────────────────────────────────
    background_tasks.add_task(process_document, doc_id, tmp_path, user_id)
    logger.info(f"Queued processing doc={doc_id}")

    return {"document_id":doc_id,"status":"pending","name":file.filename,
            "message":"Document uploaded. Processing started in background."}


@router.get("/", response_model=DocumentListOut)
async def list_documents(user=Depends(get_current_user)):
    # RLS: "Users can manage their own documents"
    docs = DB.list_documents(user["id"], user["jwt"])
    return DocumentListOut(documents=docs, total=len(docs))


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: str, user=Depends(get_current_user)):
    doc = DB.get_document(doc_id, user["jwt"])
    if not doc: raise HTTPException(404, "Document not found")
    return doc


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user=Depends(get_current_user)):
    doc = DB.get_document(doc_id, user["jwt"])
    if not doc: raise HTTPException(404, "Document not found")
    _del_pinecone(user["id"], doc_id)
    try:
        get_service_client().storage.from_(settings.SUPABASE_BUCKET).remove([doc["storage_path"]])
    except: pass
    DB.delete_document(doc_id, user["jwt"])   # CASCADE removes chunks
    return {"message":"Document deleted","document_id":doc_id}


@router.post("/{doc_id}/reindex", status_code=202)
async def reindex_document(doc_id: str, background_tasks: BackgroundTasks, user=Depends(get_current_user)):
    doc = DB.get_document(doc_id, user["jwt"])
    if not doc: raise HTTPException(404,"Document not found")
    os.makedirs(settings.TMP_DIR, exist_ok=True)
    tmp_path = f"{settings.TMP_DIR}/reindex_{doc_id}.pdf"
    data = get_service_client().storage.from_(settings.SUPABASE_BUCKET).download(doc["storage_path"])
    with open(tmp_path,"wb") as fh: fh.write(data)
    _del_pinecone(user["id"], doc_id)
    DB.delete_chunks_by_document(doc_id)
    DB.update_document_status(doc_id,"pending")
    background_tasks.add_task(process_document, doc_id, tmp_path, user["id"])
    return {"message":"Re-indexing started","document_id":doc_id}
