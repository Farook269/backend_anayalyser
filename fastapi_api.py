# fastapi_api.py
import os
import shutil
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from backend import get_embeddings
from backend import create_pdf_retriever, generate_answer, llm

#  Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()



@app.on_event("startup")
def warm_up_embeddings():
    def _load():
        import time
        time.sleep(2)  # Allow container to pass initial cloud health checks
        try:
            emb = get_embeddings()
            emb.embed_query("warmup document vectorizer")
            print("[INFO] FastEmbed model downloaded, ONNX session warmed up")
        except Exception as e:
            print(f"[WARN] Embedding warmup error: {e}")
    threading.Thread(target=_load, daemon=True).start()

# Enable CORS for Postman and frontend use
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

retriever = None  # Global retriever object (in-memory)


class ChatRequest(BaseModel):
    question: str


# 🩺 Health Check Endpoint
@app.get("/")
def home():
    return {"message": "✅ FastAPI RAG Chatbot API is running!"}


#  Upload PDF (Override Mode)
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a new PDF file.
    Deletes old files and creates a new retriever.
    """
    global retriever

    if file.filename == "":
        raise HTTPException(status_code=400, detail=" No file selected.")

    #  Delete old PDFs safely
    import gc
    gc.collect()
    for old_file in os.listdir(UPLOAD_DIR):
        if old_file != file.filename:
            try:
                os.remove(os.path.join(UPLOAD_DIR, old_file))
            except Exception as e:
                print(f"[INFO] Cleaned old file notice: {old_file} ({e})")

    # 💾 Save new PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🧠 Build retriever in background threadpool (non-blocking)
    import asyncio
    retriever = await asyncio.to_thread(create_pdf_retriever, [file_path])

    return {"status": "success", "file_uploaded": file.filename}



@app.get("/health")
def health():
    return {"status": "ok"}
# -----------------------------
# 💬 Chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """
    Chat endpoint for real-time document intelligence and live web research.
    """
    global retriever

    try:
        question = request.question
        answer = await asyncio.to_thread(generate_answer, question, retriever)
        return {"question": question, "answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "question": request.question,
            "answer": f"⚠️ An error occurred: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("fastapi_api:app", host=host, port=port)