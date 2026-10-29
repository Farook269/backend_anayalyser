import os
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.embeddings import get_embeddings

app = FastAPI(
    title="Enterprise Document Intelligence RAG",
    description="Offline-first, strictly grounded Document Intelligence RAG API powered by Ollama and ONNX FastEmbed.",
    version="1.0.0",
)


@app.on_event("startup")
def warm_up_embeddings():
    """Warms up the ONNX embedding engine asynchronously during startup."""
    def _load():
        import time
        time.sleep(1)
        try:
            emb = get_embeddings()
            emb.embed_query("warmup document vectorizer")
            print("[INFO] FastEmbed model verified, ONNX session warmed up.")
        except Exception as e:
            print(f"[WARN] Embedding warmup error: {e}")

    threading.Thread(target=_load, daemon=True).start()


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-analyser-frontend.vercel.app",
        "https://rag-analyser-frontend-zlht.onrender.com",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)

