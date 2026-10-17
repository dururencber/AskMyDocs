from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import AskRequest, AskResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/rag", tags=["RAG Question & Answer"])


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Dökümanlara Soru Sor (Cosine Similarity Search + Grounded Generation)",
)
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
) -> AskResponse:
    """
    1. Soruyu `text-embedding-3-small` ile 768 boyutlu vektöre dönüştürür.
    2. PostgreSQL `pgvector` `<=>` kosinüs mesafesi operatörüyle en alakalı parçaları bulur.
    3. Benzerlik eşiği (`similarity_threshold`) altında kalan alakasız parçaları eler.
    4. Parçaları kaynak döküman/sayfa bilgisiyle birleştirerek `Context` inşa eder.
    5. OpenAI `gpt-4o-mini` modeline katı sistem promptu ile göndererek cevabı ve kaynakları döner.
    """
    rag_service = RAGService(db=db)
    return rag_service.ask(payload)
