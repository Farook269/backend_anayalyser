"""
Backward-compatibility bridge for legacy imports.
Re-exports core RAG components from modular `app.core`.
"""
from app.core.embeddings import get_embeddings, GeminiRESTEmbeddings, LightweightFastEmbed
from app.core.llm import llm, get_vision_llm
from app.core.vectorstore import create_pdf_retriever
from app.core.rag_engine import generate_answer

__all__ = [
    "get_embeddings",
    "GeminiRESTEmbeddings",
    "LightweightFastEmbed",
    "llm",
    "get_vision_llm",
    "create_pdf_retriever",
    "generate_answer",
]
