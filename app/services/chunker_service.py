from dataclasses import dataclass
from typing import List, Optional

from app.core.config import settings
from app.services.parser_service import ExtractedPage


@dataclass
class TextChunk:
    """Parçalanmış metin birimini ve元 (meta) verisini temsil eder."""

    content: str
    chunk_index: int
    page_number: Optional[int] = None


class RecursiveChunkerService:
    """
    Anlamsal bütünlüğü koruyan Recursive Character Text Splitter implementasyonu.
    Önce paragraflardan (\n\n), sonra satırlardan (\n), ardından cümle sonlarından (. ! ?)
    ve en son kelimelerden bölerek `chunk_size` (varsayılan 500) ve `chunk_overlap` (varsayılan 50)
    kurallarına uygun parçalar üretir.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap değeri chunk_size değerinden küçük olmalıdır.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def chunk_pages(self, pages: List[ExtractedPage]) -> List[TextChunk]:
        """Sayfa listesini alır ve sıralı `TextChunk` nesneleri döner."""
        all_chunks: List[TextChunk] = []
        current_chunk_idx = 0

        for page in pages:
            page_splits = self.split_text(page.text)
            for split_content in page_splits:
                cleaned = split_content.strip()
                if cleaned:
                    all_chunks.append(
                        TextChunk(
                            content=cleaned,
                            chunk_index=current_chunk_idx,
                            page_number=page.page_number,
                        )
                    )
                    current_chunk_idx += 1

        return all_chunks

    def split_text(self, text: str) -> List[str]:
        """Metni hiyerarşik ayraçlar kullanarak rekürsif olarak parçalar."""
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []

        # Kullanılacak en uygun ayracı seç
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Ayraca göre metni böl (noktalama ayraçlarında ayracı koru)
        if separator:
            raw_splits = self._split_keep_separator(text, separator)
        else:
            raw_splits = list(text)

        good_splits: List[str] = []
        for s in raw_splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    # En alt seviyeye (karakter) inildiyse doğrudan pencereli böl
                    final_chunks.extend(self._hard_window_split(s))
                else:
                    other_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    @staticmethod
    def _split_keep_separator(text: str, separator: str) -> List[str]:
        """Metni ayraca göre bölerken ayracı önceki parçanın sonunda tutar (cümle/paragraf bütünlüğü için)."""
        parts = text.split(separator)
        result: List[str] = []
        for idx, part in enumerate(parts):
            if idx < len(parts) - 1:
                combined = part + separator
            else:
                combined = part
            if combined:
                result.append(combined)
        return result

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Küçük parçaları `chunk_size` sınırına kadar birleştirir ve `chunk_overlap` uygular."""
        merged_chunks: List[str] = []
        current_parts: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            if current_length + split_len > self.chunk_size and current_parts:
                chunk_str = "".join(current_parts).strip()
                if chunk_str:
                    merged_chunks.append(chunk_str)

                # Overlap sınırına inene kadar baştan eleman çıkar
                while current_parts and (
                    current_length > self.chunk_overlap
                    or (current_length + split_len > self.chunk_size and current_length > 0)
                ):
                    removed = current_parts.pop(0)
                    current_length -= len(removed)

            current_parts.append(split)
            current_length += split_len

        if current_parts:
            chunk_str = "".join(current_parts).strip()
            if chunk_str:
                merged_chunks.append(chunk_str)

        return merged_chunks

    def _hard_window_split(self, text: str) -> List[str]:
        """Hiçbir ayraç bulunamayan çok uzun blokları overlap'li sabit pencereyle böler."""
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: List[str] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + self.chunk_size >= len(text):
                break
        return chunks
