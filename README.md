# AskMyDocs - Mini RAG (Retrieval-Augmented Generation) API

**Python (FastAPI)**, **PostgreSQL + pgvector (HNSW Cosine Index)** ve **OpenAI (`text-embedding-3-small` & `gpt-4o-mini`)** kullanılarak sıfırdan geliştirilmiş, uçtan uca çalışan Doküman Soru-Cevap (RAG) servisi.

---

## Proje Klasör Yapısı

```text
AskMyDocs/
├── docker-compose.yml            # pgvector/pgvector:pg16 servisi
├── requirements.txt              # Python bağımlılıkları
├── .env.example                  # Ortam değişkenleri şablonu
├── sample_company_policy.txt     # Test için örnek döküman
└── app/
    ├── __init__.py
    ├── main.py                   # FastAPI uygulama giriş noktası ve lifespan DB init
    ├── core/
    │   ├── __init__.py
    │   ├── config.py             # Ortam değişkenleri ve RAG hiperparametreleri
    │   └── database.py           # SQLAlchemy engine, get_db ve HNSW index kurulumu
    ├── models/
    │   ├── __init__.py
    │   ├── domain.py             # Document ve DocumentChunk SQLAlchemy modelleri
    │   └── schemas.py            # Pydantic Request/Response şemaları
    ├── services/
    │   ├── __init__.py
    │   ├── parser_service.py     # PDF (pypdf) & TXT metin çıkarma ve temizleme
    │   ├── chunker_service.py    # Recursive Character Text Splitter (500/50)
    │   ├── embedding_service.py  # OpenAI text-embedding-3-small (1536d) servisi
    │   └── rag_service.py        # pgvector (<=>) arama, Context inşası ve gpt-4o-mini
    └── api/
        ├── __init__.py
        ├── documents.py          # POST /api/documents/upload, GET & DELETE /api/documents
        └── rag.py                # POST /api/rag/ask
```

---

## Adım Adım Çalıştırma Kılavuzu

### 1. Ortam Değişkenlerini Hazırlayın
```bash
cp .env.example .env
```
`.env` dosyasını açarak kendi `OPENAI_API_KEY` anahtarınızı girin.

### 2. PostgreSQL + pgvector Veritabanını Başlatın
```bash
docker compose up -d
```

### 3. Sanal Ortam (Virtual Environment) Kurulumu ve Paketler
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 4. FastAPI Sunucusunu Başlatın
Uygulama ilk ayağa kalktığında `CREATE EXTENSION IF NOT EXISTS vector;`, `documents` / `document_chunks` tabloları ve `HNSW (vector_cosine_ops)` indeksi otomatik olarak oluşturulur:
```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Swagger UI:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

---

## Örnek `curl` Sorguları

### 1. Doküman Yükleme (`POST /api/documents/upload`)
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_company_policy.txt"
```

### 2. Tüm Dökümanlarda Soru-Cevap (`POST /api/rag/ask`)
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Yazılım lisansı iade süresi kaç gündür ve iade şartı nedir?",
    "top_k": 3,
    "similarity_threshold": 0.30
  }'
```

### 3. Belirli Bir Dökümanla Sınırlandırılmış Soru-Cevap (`document_id` Filtreli)
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Yıllık eğitim bütçesi ne kadardır?",
    "document_id": "BURAYA_DOCUMENT_UUID_YAZIN",
    "top_k": 5
  }'
```

### 4. Bağlamda Olmayan Bir Soru (Halüsinasyon Önleme Testi)
```bash
curl -X POST "http://localhost:8000/api/rag/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Şirketin yemek kartı günlük limiti kaç TLdir?"
  }'
```
