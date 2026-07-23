# fastapi_api.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from backend import get_embeddings
from backend import create_pdf_retriever, generate_answer

#  Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()



@app.on_event("startup")
def warm_up_embeddings():
    def _load():
        get_embeddings()
        print("✅ Embedding model warmed up")
    threading.Thread(target=_load, daemon=True).start()

# Enable CORS for Postman and frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    #  Delete old PDFs
    for old_file in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, old_file))
        except Exception as e:
            print(f"⚠️ Could not delete {old_file}: {e}")

    # 💾 Save new PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🧠 Build retriever
    retriever = create_pdf_retriever([file_path])

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
    Ask a question about the uploaded PDF.
    """
    global retriever

    if retriever is None:
        raise HTTPException(status_code=400, detail="❌ No PDF uploaded yet. Please upload a file first.")

    question = request.question
    answer = generate_answer(question, retriever)

    return {"question": question, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_api:app", host="127.0.0.1", port=8000, reload=True)