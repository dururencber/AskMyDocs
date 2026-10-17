from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.core.config import settings

engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initializes PostgreSQL pgvector extension, creates tables,
    and ensures the HNSW cosine similarity index exists.
    """
    from app.models.domain import Document, DocumentChunk  # noqa: F401

    with engine.begin() as conn:
        # 1. pgvector eklentisini aktifleştir
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # 2. Tabloları ve SQLAlchemy üzerinde tanımlı indeksleri oluştur
    Base.metadata.create_all(bind=engine)

    # 3. HNSW kosinüs indeksini garanti altına al
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        )
