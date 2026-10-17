from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.documents import router as documents_router
from app.api.rag import router as rag_router
from app.core.config import settings
from app.core.database import init_db

BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlangıcında pgvector eklentisini, tabloları ve HNSW indeksini hazırlar."""
    init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "FastAPI, PostgreSQL + pgvector (HNSW Cosine Index) ve OpenAI "
        "(text-embedding-3-small & gpt-4o-mini) tabanlı uçtan uca Mini RAG API."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(rag_router)


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Kök dizinde modern koyu temalı AskMyDocs Frontend arayüzünü (index.html) sunar."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Servis sağlık kontrolü."""
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
    }
