import requests
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from app.config import USE_LOCAL_LLM, GEMINI_API_KEY


class GeminiRESTEmbeddings(Embeddings):
    """
    Direct REST embedding client for Google Gemini API.
    Bypasses SDK version conflicts, consumes 0% container CPU, and embeds chunks via Google GPU in ~0.5s.
    """
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents?key={api_key}"

    def embed_documents(self, texts):
        if not texts:
            return []
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "requests": [
                    {
                        "model": self.model,
                        "content": {"parts": [{"text": t}]}
                    }
                    for t in batch
                ]
            }
            res = requests.post(self.url, json=payload, timeout=30)
            if res.status_code != 200:
                raise ValueError(f"Gemini Embedding API Error ({res.status_code}): {res.text}")
            data = res.json()
            all_embeddings.extend([e["values"] for e in data["embeddings"]])
        return all_embeddings

    def embed_query(self, text):
        res = self.embed_documents([text])
        return res[0]


class LightweightFastEmbed(Embeddings):
    """
    Ultra-lightweight ONNX-powered embedding model (FastEmbed).
    Consumes ~40MB RAM (vs 500MB+ for PyTorch sentence-transformers),
    fitting easily within constrained CPU environments.
    """
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [list(e) for e in self.model.embed(texts)]

    def embed_query(self, text):
        return list(next(self.model.embed([text])))


_embeddings = None


def get_embeddings():
    """Singleton provider for embeddings to avoid reloading weights into memory."""
    global _embeddings
    if _embeddings is None:
        if USE_LOCAL_LLM:
            print("[INFO] Using 100% Local FastEmbed ONNX Embeddings (BAAI/bge-small-en-v1.5, 0 cloud calls)")
            _embeddings = LightweightFastEmbed(model_name="BAAI/bge-small-en-v1.5")
        elif GEMINI_API_KEY:
            print("[INFO] Using Google Gemini Cloud Embeddings via REST (instant, 0% container CPU)")
            _embeddings = GeminiRESTEmbeddings(api_key=GEMINI_API_KEY)
        else:
            print("[INFO] Using local LightweightFastEmbed fallback")
            _embeddings = LightweightFastEmbed(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings

