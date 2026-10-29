import os
import shutil
import asyncio
import gc
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.config import UPLOAD_DIR
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, UploadResponse
from app.core.vectorstore import create_pdf_retriever
from app.core.rag_engine import generate_answer

router = APIRouter()

# Global in-memory active document retriever
retriever = None


@router.get("/", response_model=dict)
def home():
    return {"message": "✅ Enterprise Document Intelligence RAG API is running!"}


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a new PDF file.
    Replaces previously uploaded files and initializes the active FAISS retriever.
    """
    global retriever

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    # Clean up old files safely to conserve disk space
    gc.collect()
    for old_file in os.listdir(UPLOAD_DIR):
        if old_file != file.filename:
            try:
                os.remove(os.path.join(UPLOAD_DIR, old_file))
            except Exception as e:
                print(f"[INFO] Cleaned old file notice: {old_file} ({e})")

    # Save new uploaded PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Build retriever in a non-blocking threadpool worker
    retriever = await asyncio.to_thread(create_pdf_retriever, [file_path])

    return {"status": "success", "file_uploaded": file.filename}


@router.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """
    Chat endpoint for strictly grounded document intelligence.
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

