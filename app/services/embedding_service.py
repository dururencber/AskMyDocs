import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

EMBEDDING_MODEL = "models/gemini-embedding-001"

def _truncate_or_pad(values: list[float], target_dim: int = 768) -> list[float]:
    if len(values) >= target_dim:
        return values[:target_dim]
    return values + [0.0] * (target_dim - len(values))

def get_embedding(text: str) -> list[float]:
    client = _get_client()
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return response.embeddings[0].values
    except Exception:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        return _truncate_or_pad(response.embeddings[0].values, 768)

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return [e.values for e in response.embeddings]
    except Exception:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        return [_truncate_or_pad(e.values, 768) for e in response.embeddings]

class EmbeddingService:
    def __init__(self):
        self.client = _get_client()

    def generate_embedding(self, text: str) -> list[float]:
        return get_embedding(text)

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings_batch(texts)

    def get_embedding(self, text: str) -> list[float]:
        return get_embedding(text)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings_batch(texts)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings_batch(texts)
