import os
import uuid
from dotenv import load_dotenv
from google import genai
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.domain import Document, DocumentChunk
from app.services.embedding_service import get_embedding

load_dotenv()

def _get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def execute_rag(db: Session, document_id: str, question: str, top_k: int = 5, similarity_threshold: float = 1.0) -> dict:
    client = _get_client()
    query_embedding = get_embedding(question)

    try:
        parsed_id = uuid.UUID(str(document_id))
    except Exception:
        parsed_id = str(document_id)

    distance_col = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(DocumentChunk, distance_col, Document.filename)
        .outerjoin(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.document_id == parsed_id)
        .order_by(distance_col)
        .limit(top_k or 5)
    )
    results = db.execute(stmt).all()

    if not results and isinstance(parsed_id, uuid.UUID):
        stmt_fallback = (
            select(DocumentChunk, distance_col, Document.filename)
            .outerjoin(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.document_id == str(document_id))
            .order_by(distance_col)
            .limit(top_k or 5)
        )
        results = db.execute(stmt_fallback).all()

    if not results:
        return {
            "question": question,
            "answer": "Belirtilen dökümana ait içerik bulunamadı.",
            "context": "",
            "sources": []
        }

    sources_list = []
    context_chunks = []

    for chunk, dist, filename in results:
        distance_val = float(dist) if dist is not None else 0.0
        similarity_val = max(0.0, 1.0 - distance_val)
        
        context_chunks.append(chunk.content)
        sources_list.append({
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "document_name": filename or "Doküman",
            "chunk_index": getattr(chunk, "chunk_index", 0),
            "content": chunk.content,
            "similarity_score": round(similarity_val, 4),
            "cosine_distance": round(distance_val, 4)
        })

    context = "\n\n".join(context_chunks)

    prompt = f"""Sen döküman analizi yapan bir yapay zeka asistanısın.
Aşağıdaki BAĞLAM (CONTEXT) bilgilerini kullanarak KULLANICI SORUSU'nu yanıtla.
Eğer sorunun cevabı bağlam içinde kesinlikle yoksa, sadece "Yüklenen dökümanda bu sorunun cevabı bulunmamaktadır." de ve tahmin yürütme.

BAĞLAM:
{context}

KULLANICI SORUSU:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return {
        "question": question,
        "answer": response.text,
        "context": context,
        "sources": sources_list
    }

class RAGService:
    def __init__(self, db: Session = None):
        self.db = db

    def ask(self, *args, **kwargs) -> dict:
        if args and hasattr(args[0], "document_id"):
            payload = args[0]
            doc_id = str(payload.document_id)
            question = payload.question
            top_k = getattr(payload, "top_k", 5)
            threshold = getattr(payload, "similarity_threshold", 1.0)
            return execute_rag(self.db, doc_id, question, top_k=top_k, similarity_threshold=threshold)

        doc_id = kwargs.get("document_id") or (args[0] if len(args) > 0 else None)
        question = kwargs.get("question") or (args[1] if len(args) > 1 else None)
        top_k = kwargs.get("top_k", 5)
        threshold = kwargs.get("similarity_threshold", 1.0)
        target_db = kwargs.get("db") or self.db

        return execute_rag(target_db, str(doc_id), question, top_k=top_k, similarity_threshold=threshold)

    def ask_question(self, *args, **kwargs) -> dict:
        return self.ask(*args, **kwargs)
