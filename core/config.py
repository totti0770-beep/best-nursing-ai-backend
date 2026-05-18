"""
core/config.py
All environment variables are loaded from a .env file.
Copy .env.example → .env and fill in your real keys.
"""

from functools import lru_cache
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Best Nursing Practice AI"
    APP_ENV: str = "development"          # development | production
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "https://your-flutterflow-app.com",
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        """Allow comma-separated string from env: 'http://a.com,http://b.com'"""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""       # server-side only — never expose to client
    SUPABASE_BUCKET: str = "clinical-docs"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 3072
    CHAT_MODEL: str = "gpt-4o"
    CHAT_MAX_TOKENS: int = 1500
    CHAT_TEMPERATURE: float = 0.1

    # ── Pinecone ──────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "nursing-ai"
    PINECONE_ENVIRONMENT: str = "gcp-starter"  # check your Pinecone dashboard

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512               # tokens per chunk
    CHUNK_OVERLAP: int = 50             # overlap between adjacent chunks
    TOP_K_CHUNKS: int = 5               # chunks retrieved per query
    MIN_RELEVANCE_SCORE: float = 0.72   # filter out low-quality matches

    # ── File Upload ───────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_MIME_TYPES: List[str] = ["application/pdf"]
    TMP_DIR: str = "/tmp/nursing_ai_uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
