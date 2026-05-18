"""
Best Nursing Practice AI — FastAPI Backend
==========================================
Entry point. Registers all routers, middleware, and startup events.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import auth, documents, chat, feedback, users
from core.database import init_db
from core.config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once at startup; clean up on shutdown."""
    logger.info("🚀 Starting Best Nursing Practice AI backend …")
    await init_db()          # create Supabase tables if missing
    logger.info("✅ Database ready")
    yield
    logger.info("🛑 Shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Best Nursing Practice AI",
    description="RAG-powered clinical assistant backend",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"

app.include_router(auth.router,      prefix=f"{PREFIX}/auth",      tags=["Auth"])
app.include_router(documents.router, prefix=f"{PREFIX}/documents",  tags=["Documents"])
app.include_router(chat.router,      prefix=f"{PREFIX}/chat",       tags=["Chat"])
app.include_router(feedback.router,  prefix=f"{PREFIX}/feedback",   tags=["Feedback"])
app.include_router(users.router,     prefix=f"{PREFIX}/users",      tags=["Users"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
