from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.domain import Document, DocumentChunk
from app.models.schemas import DocumentListItem, DocumentUploadResponse
from app.services.chunker_service import RecursiveChunkerService
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import DocumentParserService

router = APIRouter(prefix="/api/documents", tags=["Documents & Ingestion"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Döküman Yükleme ve Vektör İndeksleme (Ingestion) Hattı",
)
async def upload_document(
    file: UploadFile = File(..., description="Yüklenecek PDF veya TXT dosyası"),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    """
    1. Kullanıcıdan PDF veya TXT dosyası alır.
    2. Dosya türüne göre düz metni (raw text) çıkarır ve temizler.
    3. Metni Recursive Chunking (`chunk_size=500`, `chunk_overlap=50`) ile parçalar.
    4. Her parça için OpenAI `text-embedding-3-small` (768 boyut) ile vektör üretir.
    5. `documents` ve `document_chunks` tablolarına ilişkisel olarak kaydeder.
    """
    file_bytes = await file.read()
    filename = file.filename or "unnamed_document.txt"

    # 1 & 2. Dosya Türü Kontrolü, Metin Çıkarma ve Temizleme
    file_type, extracted_pages = DocumentParserService.extract_pages(
        file_bytes=file_bytes,
        filename=filename,
    )
    total_characters = sum(len(page.text) for page in extracted_pages)

    # 3. Recursive Chunking (chunk_size=500, chunk_overlap=50)
    chunker = RecursiveChunkerService()
    text_chunks = chunker.chunk_pages(extracted_pages)

    if not text_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Döküman parçalanamadı veya geçerli metin bloğu üretilemedi.",
        )

    # 4. OpenAI text-embedding-3-small (768 boyut) ile Vektör Üretimi
    embedding_service = EmbeddingService()
    chunk_texts = [c.content for c in text_chunks]
    embeddings = embedding_service.generate_embeddings_batch(chunk_texts)

    # 5. Veritabanına İlişkisel Kayıt (Transaction)
    try:
        new_doc = Document(
            filename=filename,
            file_type=file_type,
        )
        db.add(new_doc)
        db.flush()  # new_doc.id UUID değerini almak için flush

        chunk_entities = [
            DocumentChunk(
                document_id=new_doc.id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                embedding=vector,
            )
            for chunk, vector in zip(text_chunks, embeddings)
        ]

        db.add_all(chunk_entities)
        db.commit()
        db.refresh(new_doc)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Veritabanına kayıt sırasında hata oluştu: {str(exc)}",
        ) from exc

    return DocumentUploadResponse(
        id=new_doc.id,
        document_id=new_doc.id,
        filename=new_doc.filename,
        file_type=new_doc.file_type,
        total_characters=total_characters,
        chunk_count=len(chunk_entities),
        total_chunks=len(chunk_entities),
        created_at=new_doc.created_at,
        message="Döküman başarıyla parçalandı, vektörlendi ve veritabanına kaydedildi.",
    )


@router.get(
    "",
    response_model=List[DocumentListItem],
    summary="Yüklü Dökümanları Listele",
)
def list_documents(db: Session = Depends(get_db)) -> List[DocumentListItem]:
    """Veritabanına yüklenmiş tüm dökümanları ve parça (chunk) sayılarını döner."""
    rows = (
        db.query(
            Document,
            func.count(DocumentChunk.id).label("chunk_count"),
        )
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
        .all()
    )

    return [
        DocumentListItem(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            chunk_count=count,
            created_at=doc.created_at,
        )
        for doc, count in rows
    ]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Dökümanı ve İlişkili Parçalarını Sil",
)
def delete_document(document_id: UUID, db: Session = Depends(get_db)) -> dict:
    """Belirtilen dökümanı ve CASCADE ile tüm chunk kayıtlarını siler."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Döküman bulunamadı: {document_id}",
        )
    db.delete(doc)
    db.commit()
    return {"message": f"'{doc.filename}' ({document_id}) ve ilişkili tüm parçaları silindi."}
