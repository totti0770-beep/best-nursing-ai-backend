"""
core/auth.py — JWT dependency + Supabase token validation.
"""
from __future__ import annotations
import logging
from typing import Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client
from core.config import settings

logger = logging.getLogger(__name__)
bearer = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> Dict:
    """
    Validate the Supabase-issued JWT by calling auth.get_user(jwt).

    Why server-side validation (not local decode)?
    ─────────────────────────────────────────────
    • Supabase handles token revocation transparently.
    • No secret key duplication across services.
    • Works with both access-tokens and refresh-tokens.

    Returns: { "id": str, "email": str, "jwt": str }
    The "jwt" key is forwarded to every DB helper so that
    get_user_client(jwt) injects it and RLS is enforced.
    """
    jwt = credentials.credentials
    try:
        client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
        resp = client.auth.get_user(jwt)
        if not resp or not resp.user:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
        u = resp.user
        return {"id": str(u.id), "email": u.email, "jwt": jwt}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"Auth failed: {exc}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Could not validate credentials")
