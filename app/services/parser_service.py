import io
import re
from dataclasses import dataclass
from typing import List, Optional
from fastapi import HTTPException, status
from pypdf import PdfReader


@dataclass
class ExtractedPage:
    """Dökümandan çıkarılan sayfa veya metin bloğunu temsil eder."""

    text: str
    page_number: Optional[int] = None


class DocumentParserService:
    """PDF ve TXT dosyalarından düz metin (raw text) çıkarma ve temizleme servisi."""

    SUPPORTED_EXTENSIONS = {"pdf", "txt"}

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Metindeki bozuk karakterleri, null byte'ları ve gereksiz boşlukları temizler;
        ancak paragraf yapısını (\n\n) koruyarak anlamsal bütünlüğü muhafaza eder.
        """
        if not raw_text:
            return ""

        # PostgreSQL TEXT kolonlarının kabul etmediği null karakterleri (\x00) temizle
        text = raw_text.replace("\x00", "")

        # Windows/Mac satır sonlarını standart \n yapısına dönüştür
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Satır sonlarında tire (-) ile bölünmüş kelimeleri birleştir (örn: "geliş-\ntirme" -> "geliştirme")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Aynı satır içindeki ardışık boşluk/tab karakterlerini tek boşluğa indirge
        text = re.sub(r"[ \t\f\v]+", " ", text)

        # Her satırın başındaki ve sonundaki boşlukları temizle
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 3 veya daha fazla ardışık yeni satırı 2 yeni satıra (paragraf ayracı) indirge
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @classmethod
    def extract_pages(cls, file_bytes: bytes, filename: str) -> tuple[str, List[ExtractedPage]]:
        """
        Dosya uzantısına göre metni çıkarır ve temizler.
        Dönüş: (file_type, List[ExtractedPage])
        """
        if not filename or "." not in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz dosya adı. Lütfen .pdf veya .txt uzantılı bir dosya yükleyin.",
            )

        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Desteklenmeyen dosya formatı: .{ext}. Sadece .pdf ve .txt dosyaları desteklenmektedir.",
            )

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Yüklenen dosya boş.",
            )

        if ext == "pdf":
            pages = cls._extract_from_pdf(file_bytes)
        else:
            pages = cls._extract_from_txt(file_bytes)

        if not pages or sum(len(p.text) for p in pages) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Dosyadan anlamlı bir metin çıkarılamadı (taranmış görüntü/boş PDF olabilir).",
            )

        return ext, pages

    @classmethod
    def _extract_from_pdf(cls, file_bytes: bytes) -> List[ExtractedPage]:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PDF dosyası okunamadı: {str(exc)}",
            ) from exc

        extracted_pages: List[ExtractedPage] = []
        for idx, page in enumerate(reader.pages, start=1):
            raw_page_text = page.extract_text() or ""
            cleaned_page_text = cls.clean_text(raw_page_text)
            if cleaned_page_text:
                extracted_pages.append(
                    ExtractedPage(text=cleaned_page_text, page_number=idx)
                )

        return extracted_pages

    @classmethod
    def _extract_from_txt(cls, file_bytes: bytes) -> List[ExtractedPage]:
        decoded_text: Optional[str] = None
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
            try:
                decoded_text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TXT dosyası karakter kodlaması çözümlenemedi.",
            )

        cleaned_text = cls.clean_text(decoded_text)
        if not cleaned_text:
            return []

        return [ExtractedPage(text=cleaned_text, page_number=1)]
