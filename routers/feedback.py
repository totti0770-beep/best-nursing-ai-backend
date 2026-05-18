"""routers/feedback.py"""
from fastapi import APIRouter, Depends, HTTPException
from core.auth import get_current_user
from core.database import DB
from models.schemas import FeedbackOut, FeedbackRequest

router = APIRouter()

@router.post("/", response_model=FeedbackOut, status_code=201)
async def submit_feedback(body: FeedbackRequest, user=Depends(get_current_user)):
    row = DB.insert_feedback({
        "message_id": body.message_id,
        "user_id":    user["id"],
        "type":       body.type,
        "reason":     body.reason,
        "comment":    body.comment,
        "question":   body.question,
        "answer":     body.answer,
    }, user["jwt"])
    return FeedbackOut(**row)

@router.get("/")
async def list_feedback(user=Depends(get_current_user)):
    """Admin-only — returns all feedback via service client."""
    rows = DB.list_feedback(limit=200)
    return {"feedback": rows, "total": len(rows)}
