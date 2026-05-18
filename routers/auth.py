"""routers/auth.py"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from core.auth import get_current_user
from core.config import settings
from core.database import DB, get_anon_client
from models.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    """
    1. Call Supabase Auth sign-up (creates auth.users row).
    2. Insert into public.users (linked by UUID FK).
    3. Return JWT + user object.
    """
    client = get_anon_client()
    try:
        resp = client.auth.sign_up({"email":body.email,"password":body.password})
    except Exception as e:
        raise HTTPException(400, f"Registration failed: {e}")

    if not resp.user:
        raise HTTPException(400, "Registration failed — check email/password")

    user_id = str(resp.user.id)
    DB.create_user({"id":user_id,"email":body.email,
                    "full_name":body.full_name,"specialty":body.specialty})

    user_row = DB.get_user(user_id, resp.session.access_token)
    return TokenResponse(
        access_token=resp.session.access_token,
        refresh_token=resp.session.refresh_token,
        user=UserOut(**user_row),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    client = get_anon_client()
    try:
        resp = client.auth.sign_in_with_password({"email":body.email,"password":body.password})
    except Exception as e:
        raise HTTPException(401, "Invalid email or password")

    if not resp.session:
        raise HTTPException(401, "Invalid credentials")

    user_row = DB.get_user(str(resp.user.id), resp.session.access_token)
    return TokenResponse(
        access_token=resp.session.access_token,
        refresh_token=resp.session.refresh_token,
        user=UserOut(**user_row),
    )


@router.post("/logout")
async def logout(user=Depends(get_current_user)):
    get_anon_client().auth.sign_out()
    return {"message":"Logged out successfully"}


@router.get("/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    row = DB.get_user(user["id"], user["jwt"])
    if not row: raise HTTPException(404,"User not found")
    return UserOut(**row)
