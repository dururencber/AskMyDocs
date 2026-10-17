from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    id: UUID = Field(..., description="Veritabanına kaydedilen dökümanın benzersiz UUID'si (id alias)")
    document_id: UUID = Field(..., description="Veritabanına kaydedilen dökümanın benzersiz UUID'si")
    filename: str = Field(..., description="Yüklenen dosyanın adı")
    file_type: str = Field(..., description="Dosya tipi (pdf veya txt)")
    total_characters: int = Field(..., description="Çıkarılan ve temizlenen toplam karakter sayısı")
    chunk_count: int = Field(..., description="Oluşturulan ve vektörlenen toplam chunk sayısı (chunk_count alias)")
    total_chunks: int = Field(..., description="Oluşturulan ve vektörlenen toplam chunk sayısı")
    created_at: datetime = Field(..., description="Kayıt oluşturulma zamanı")
    message: str = Field(..., description="İşlem durumu mesajı")

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    id: UUID
    filename: str
    file_type: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Dökümanlara sorulacak soru metni",
        examples=["İade politikası kaç gündür ve şartları nelerdir?"],
    )
    document_id: Optional[UUID] = Field(
        default=None,
        description="Opsiyonel: Aramayı yalnızca belirli bir dökümanla sınırlandırmak için döküman UUID'si",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Kosinüs benzerlik aramasında çekilecek maksimum parça (chunk) sayısı (varsayılan: 5)",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Opsiyonel: Minimum kosinüs benzerlik skoru eşiği (0.0 - 1.0). Belirtilmezse varsayılan (0.30) kullanılır.",
    )


class SourceChunkResponse(BaseModel):
    chunk_id: UUID = Field(..., description="Parçanın (chunk) benzersiz ID'si")
    document_id: UUID = Field(..., description="Parçanın ait olduğu dökümanın ID'si")
    document_name: str = Field(..., description="Parçanın ait olduğu dökümanın dosya adı")
    chunk_index: int = Field(..., description="Parçanın döküman içindeki sıra numarası")
    page_number: Optional[int] = Field(
        default=None, description="PDF dökümanlar için parçanın bulunduğu sayfa numarası"
    )
    similarity_score: float = Field(..., description="Soru ile parça arasındaki kosinüs benzerlik skoru (0-1)")
    cosine_distance: float = Field(..., description="PostgreSQL pgvector <=> kosinüs mesafesi")
    content: str = Field(..., description="Parçanın metin içeriği")


class AskResponse(BaseModel):
    question: str = Field(..., description="Kullanıcının sorduğu soru")
    answer: str = Field(..., description="LLM (gpt-4o-mini) tarafından sadece bağlama dayanarak üretilen cevap")
    context: str = Field(..., description="LLM'e gönderilen birleştirilmiş tam bağlam (Context) metni")
    sources: List[SourceChunkResponse] = Field(
        default_factory=list,
        description="Cevap üretilirken kullanılan kaynak chunk metinleri, skorları ve döküman bilgileri",
    )
