"""routers/users.py"""
from fastapi import APIRouter, Depends, HTTPException
from core.auth import get_current_user
from core.database import DB
from models.schemas import UserOut, UserSettingsUpdate

router = APIRouter()

@router.get("/settings", response_model=UserOut)
async def get_settings(user=Depends(get_current_user)):
    row = DB.get_user(user["id"], user["jwt"])
    if not row: raise HTTPException(404,"User not found")
    return UserOut(**row)

@router.patch("/settings", response_model=UserOut)
async def update_settings(body: UserSettingsUpdate, user=Depends(get_current_user)):
    """Partial update — only include fields that changed."""
    payload = {}
    if body.full_name   is not None: payload["full_name"]  = body.full_name
    if body.specialty   is not None: payload["specialty"]  = body.specialty
    if body.language    is not None or body.detail_level is not None:
        row = DB.get_user(user["id"], user["jwt"]) or {}
        current = row.get("settings",{})
        if body.language    is not None: current["language"]     = body.language
        if body.detail_level is not None: current["detail_level"] = body.detail_level
        payload["settings"] = current
    if not payload: raise HTTPException(400,"No fields to update")
    updated = DB.update_user(user["id"], payload, user["jwt"])
    return UserOut(**updated)

@router.delete("/data", status_code=200)
async def delete_all_data(user=Depends(get_current_user)):
    """GDPR right-to-deletion. Cascades through all tables."""
    from core.database import get_service_client
    svc = get_service_client()
    # Delete all documents → CASCADE: chunks, messages, sessions, feedback
    docs = DB.list_documents(user["id"], user["jwt"])
    for d in docs:
        try:
            from core.config import settings as cfg
            svc.storage.from_(cfg.SUPABASE_BUCKET).remove([d["storage_path"]])
        except: pass
    svc.table("users").delete().eq("id", user["id"]).execute()
    return {"message":"All user data deleted"}
