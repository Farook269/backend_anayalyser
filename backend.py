import os
import base64

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1️⃣ Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file.")
if not SERPER_API_KEY:
    raise ValueError("❌ SERPER_API_KEY not found in .env file.")

# 2️⃣ Lightweight FastEmbed Embeddings (<50MB RAM, ONNX runtime, no PyTorch)
class LightweightFastEmbed(Embeddings):
    """
    Ultra-lightweight ONNX-powered embedding model (FastEmbed).
    Consumes ~40MB RAM (vs 500MB+ for PyTorch sentence-transformers),
    fitting easily within Render 512MB free tier containers.
    """
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [list(e) for e in self.model.embed(texts)]

    def embed_query(self, text):
        return list(next(self.model.embed([text])))

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = LightweightFastEmbed(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings



 # Step 2: Initialize Groq LLM
 



# 3️⃣ Model names — both free-tier on Groq (rate-limited, no card required)
TEXT_MODEL = "qwen/qwen3.8-27b"
VISION_MODEL = "qwen/qwen3.8-27b"

 # Step 2: Initialize Groq LLM (bounded output tokens to satisfy Groq 1000 OTPM limit)
llm = ChatGroq(
        model=TEXT_MODEL,
        temperature=0.3,
        max_tokens=600,
        groq_api_key=GROQ_API_KEY
)

# -----------------------------
# 🖼️ Vision-based page analysis (for scanned pages / diagrams / flowcharts)
# -----------------------------
# Shared vision LLM instance (lazy initialized)
_vision_llm = None

def get_vision_llm():
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = ChatGroq(
            model=VISION_MODEL,
            temperature=0.2,
            max_tokens=500,
            groq_api_key=GROQ_API_KEY,
        )
    return _vision_llm


def _page_to_base64(pdf_path, page_number, zoom=1.2):
    """Render a single PDF page to a base64 PNG string with optimized resolution."""
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def _describe_page_with_vision(pdf_path, page_number):
    """
    Ask the vision LLM to read and describe a page's visual content,
    including any diagrams, flowcharts, or images.
    """
    b64_image = _page_to_base64(pdf_path, page_number)
    vision_llm = get_vision_llm()

    prompt = """Extract all information from this document page.

    - Transcribe all readable text exactly.
    - If there's a flowchart, diagram, or process flow: describe each step/box
      in order, and describe the arrows/connections between them (e.g.
      "Box A leads to Box B, which branches into Box C and Box D").
    - If there's a chart, table, or graph: describe what it shows.
    - Be thorough and factual — do not skip visual details."""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            },
        ]
    )

    response = vision_llm.invoke([message])
    return response.content.strip()


def extract_documents_with_vision(path, min_text_len=30):
    """
    High-performance text extraction using PyMuPDF (fitz).
    Natively parses pages in milliseconds. Vision is ONLY invoked
    if a page contains virtually no extractable text AND contains an embedded image
    (e.g. scanned document, pure diagram, or infographic).
    """
    doc = fitz.open(path)
    page_texts = {}
    pages_with_images = set()

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()
        page_texts[i] = text
        if len(page.get_images()) > 0:
            pages_with_images.add(i)

    doc.close()

    # Only call Vision LLM on pages that both lack native text AND actually have images
    needs_vision = [
        i for i, text in page_texts.items()
        if len(text) < min_text_len and i in pages_with_images
    ]

    vision_results = {}
    if needs_vision:
        print(f"[INFO] Running vision extraction on {len(needs_vision)} visual/scanned page(s): {needs_vision}")
        with ThreadPoolExecutor(max_workers=min(3, len(needs_vision))) as executor:
            future_to_page = {
                executor.submit(_describe_page_with_vision, path, i): i
                for i in needs_vision
            }
            for future in as_completed(future_to_page):
                i = future_to_page[future]
                try:
                    vision_results[i] = future.result()
                except Exception as e:
                    print(f"[WARN] Vision extraction failed on page {i}: {e}")

    docs = []
    for i, text in page_texts.items():
        if i in vision_results:
            text = f"{text}\n{vision_results[i]}".strip()

        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page": i},
                )
            )

    return docs


# -----------------------------
# 📚 PDF retriever setup
# -----------------------------
def create_pdf_retriever(file_paths):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunk_id = 0

    for path in file_paths:
        docs = extract_documents_with_vision(path)
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["chunk"] = chunk_id
            chunk_id += 1
        all_docs.extend(chunks)

    if not all_docs:
        raise ValueError(
            "❌ No extractable text found in the uploaded PDF(s), even after vision analysis. "
            "The file may be corrupted or contain no readable content."
        )

    vectorstore = FAISS.from_documents(all_docs, get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


# -----------------------------
# 🌐 Serper API — fetch live web data (Google Search)
# -----------------------------
def fetch_live_data_from_serper(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    snippets = []
    if "organic" in data:
        for item in data["organic"]:
            if "snippet" in item:
                snippets.append(item["snippet"])

    if not snippets:
        return "No relevant live data found."

    return "\n".join(snippets[:5])  # use top 5 snippets


# -----------------------------
# 💬 Combine RAG + Serper + General fallback
# -----------------------------
def generate_answer(user_question, retriever, include_sources=True):
    # Step 1: Detect if the user is asking an overview/meta question about the uploaded document
    q_lower = user_question.lower().strip()
    doc_keywords = [
        "what is the pdf", "what is this pdf", "about the pdf", "summarize", "summary",
        "overview", "explain the pdf", "explain me about the pdf", "explain me aobut the pdf",
        "explain this pdf", "about this document", "about the document", "who is this",
        "who is the person", "tell me about this", "what is in this", "what is this document",
        "what is this file", "describe the pdf", "describe this document"
    ]
    is_doc_meta_query = any(k in q_lower for k in doc_keywords)

    # Retrieve relevant chunks (for broad overview questions, query broadly to capture full context)
    if is_doc_meta_query:
        relevant_docs = retriever.invoke("summary overview introduction profile background experience skills content")
    else:
        relevant_docs = retriever.invoke(user_question)

    # Fallback to direct query if broad retrieval returned empty
    if not relevant_docs:
        relevant_docs = retriever.invoke(user_question)

    context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
    sources = []
    for doc in relevant_docs:
        meta = doc.metadata or {}
        sources.append({
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 0),
        })

    # Step 2: Synthesize answer from context
    if context.strip():
        prompt = f"""You are a professional, accurate document assistant analyzing an uploaded document.
The user's question is: "{user_question}"

Instructions:
1. "The PDF", "the document", or "this file" refers to the uploaded document represented in the Context below.
2. If the user asks for a summary or "what is the pdf about", synthesize the context and provide a structured, thorough overview of who or what the document is about, key skills/topics, projects, and main details.
3. Base your answer strictly on the provided Context.
4. Format your answer in clean Markdown: use **bold** for key terms, names, and metrics. Use clear bullet points and sub-headings (##) if multiple topics are covered.
5. ONLY if the Context contains absolutely no relevant information to answer the question, respond with: "Not found in PDFs."

Context:
{context}
"""
        response = llm.invoke(prompt)
        answer_text = response.content

        # Step 3: Handle Fallback
        if "Not found in PDFs" in answer_text:
            if is_doc_meta_query:
                # Do NOT search Google for "what is the pdf about" — that would search for the Adobe PDF file format!
                answer_text = (
                    "The uploaded document does not appear to contain sufficient information to answer that specific question. "
                    "Please ask a more specific question about the document's content."
                )
                sources = []
            else:
                print("🔍 Falling back to live Serper search...")
                answer_text = fetch_live_answer_with_serper(user_question, llm)
                sources = []
    else:
        if is_doc_meta_query:
            answer_text = "No extractable text found in the uploaded document. Please check the file."
            sources = []
        else:
            print("🔍 No PDF context found — using live Serper search...")
            answer_text = fetch_live_answer_with_serper(user_question, llm)
            sources = []

    # Step 6: Format sources for UI if requested
    if include_sources:
        if sources:
            # Deduplicate by (source, page) so the same page isn't listed 4x for 4 chunks
            seen = set()
            unique_sources = []
            for s in sources:
                key = (s["source"], s["page"])
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)

            source_lines = "\n".join([
                f"- {s['source']} (page {s['page'] + 1})" for s in unique_sources
            ])
            source_md = f"\n\n---\n### Sources\n{source_lines}"
        else:
            source_md = "\n\n---\n### Sources\nGenerated by LLM (no PDF match found)"

        answer_text = f"{answer_text}{source_md}"

    return answer_text


# Dedicated live search LLM instance (compact token budget to prevent OTPM 429)
_serper_llm = None

def get_serper_llm():
    global _serper_llm
    if _serper_llm is None:
        _serper_llm = ChatGroq(
            model=TEXT_MODEL,
            temperature=0.2,
            max_tokens=250,
            groq_api_key=GROQ_API_KEY,
        )
    return _serper_llm


# Use Serper API + LLM for live answers
def fetch_live_answer_with_serper(question, llm=None):
    try:
        serper_context = fetch_live_data_from_serper(question)
        search_model = get_serper_llm()

        prompt = f"""You are a factual assistant. Using the following live web search results,
extract the most accurate and up-to-date answer to the question concisely in markdown.
Do not call any tools or functions. Output plain text directly.

Context:
{serper_context}

Question:
{question}
"""
        response = search_model.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[WARN] Live search fallback failed: {e}")
        return f"Could not retrieve live information for '{question}'. (Error: {str(e)})"


# ================================================================
# 7️⃣ Example usage
# ================================================================
# if __name__ == "__main__":
#     pdf_files = ["data/sample1.pdf"]  # Change to your actual PDF paths
#     retriever = create_pdf_retriever(pdf_files)

#     print("✅ System ready! You can now ask live + RAG questions.")
#     while True:
#         query = input("\n🧠 Ask me anything: ")
#         reply = generate_answer(query, retriever)
#         print("\n🤖", reply)


