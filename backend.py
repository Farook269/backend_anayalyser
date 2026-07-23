import os
import base64
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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

# 2️⃣ Initialize Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3️⃣ Model names — both free-tier on Groq (rate-limited, no card required)
TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"


# -----------------------------
# 🖼️ Vision-based page analysis (for scanned pages / diagrams / flowcharts)
# -----------------------------
def _page_to_base64(pdf_path, page_number, zoom=2.0):
    """Render a single PDF page to a base64 PNG string."""
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

    vision_llm = ChatGroq(
        model=VISION_MODEL,
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
    )

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


def _page_has_image(pdf_path, page_number):
    """Check whether a page actually contains an embedded image/diagram."""
    doc = fitz.open(pdf_path)
    has_image = len(doc[page_number].get_images(full=True)) > 0
    doc.close()
    return has_image


def extract_documents_with_vision(path, min_text_len=40):
    """
    Extract text per page. Vision is only invoked for pages that BOTH have
    little native text AND contain an embedded image — this avoids wasting
    a vision call on normal text pages that PyPDFLoader under-extracts
    (e.g. resumes with tight column layouts). Vision calls run in parallel
    to avoid stacking up upload latency page-by-page.
    """
    loader = PyPDFLoader(path)
    pages = loader.load()

    page_texts = {i: (p.page_content or "") for i, p in enumerate(pages)}
    needs_vision = [
        i for i, text in page_texts.items()
        if len(text.strip()) < min_text_len and _page_has_image(path, i)
    ]

    vision_results = {}
    if needs_vision:
        with ThreadPoolExecutor(max_workers=min(4, len(needs_vision))) as executor:
            future_to_page = {
                executor.submit(_describe_page_with_vision, path, i): i
                for i in needs_vision
            }
            for future in as_completed(future_to_page):
                i = future_to_page[future]
                try:
                    vision_results[i] = future.result()
                except Exception as e:
                    print(f"⚠️ Vision extraction failed on page {i}: {e}")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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

    vectorstore = FAISS.from_documents(all_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
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
    # Step 1: Retrieve document context
    relevant_docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
    sources = []
    for doc in relevant_docs:
        meta = doc.metadata or {}
        sources.append({
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 0),
        })

    # Step 2: Initialize Groq LLM
    llm = ChatGroq(
        model=TEXT_MODEL,
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )

    # Step 3: If context found, use it
    if context.strip():
        prompt = f"""
            You are a professional assistant helping answer questions using document context.
            Write a clear, well-organized answer in your own words — do not copy raw text or
            bullet-dump the context verbatim. Synthesize the information into a natural,
            readable response, using short paragraphs or clean bullet points only where it
            genuinely helps clarity.

            Format the answer in Markdown: **bold** key terms, names, numbers, and important
            facts. Use bullet points or numbered lists when listing multiple items (e.g. skills,
            dates, steps). Use short sub-headings (##) if the answer covers more than one topic.

            If the answer is not present in the context, respond only with: "Not found in PDFs."

            Context:
            {context}

            Question:
            {user_question}
            """
        response = llm.invoke(prompt)
        answer_text = response.content

        # Step 4: Fallback to Serper if PDF doesn't help
        if "Not found in PDFs" in answer_text:
            print("🔍 Falling back to live Serper search...")
            answer_text = fetch_live_answer_with_serper(user_question, llm)
            sources = []
    else:
        # Step 5: If no PDF context, go directly to Serper
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


# Use Serper API + LLM for live answers
def fetch_live_answer_with_serper(question, llm):
    serper_context = fetch_live_data_from_serper(question)

    prompt = f"""
    You are a factual assistant. Using the following live web search results,
    extract the **most accurate and up-to-date information** related to the question.

    Context:
    {serper_context}

    Question:
    {question}

    Only return the factual answer, not explanation.
    """
    response = llm.invoke(prompt)
    return response.content.strip()


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









# import os
# import base64
# import requests
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage
# import fitz  # PyMuPDF

# # 1️⃣ Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError("❌ GROQ_API_KEY not found in .env file.")
# if not SERPER_API_KEY:
#     raise ValueError("❌ SERPER_API_KEY not found in .env file.")

# # 2️⃣ Initialize Embedding model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # 3️⃣ Model names — both free-tier on Groq (rate-limited, no card required)
# TEXT_MODEL = "openai/gpt-oss-20b"
# VISION_MODEL = "qwen/qwen3.6-27b"


# # -----------------------------
# # 🖼️ Vision-based page analysis (for scanned pages / diagrams / flowcharts)
# # -----------------------------
# def _page_to_base64(pdf_path, page_number, zoom=2.0):
#     """Render a single PDF page to a base64 PNG string."""
#     doc = fitz.open(pdf_path)
#     page = doc[page_number]
#     mat = fitz.Matrix(zoom, zoom)
#     pix = page.get_pixmap(matrix=mat)
#     img_bytes = pix.tobytes("png")
#     doc.close()
#     return base64.b64encode(img_bytes).decode("utf-8")


# def _describe_page_with_vision(pdf_path, page_number):
#     """
#     Ask the vision LLM to read and describe a page's visual content,
#     including any diagrams, flowcharts, or images.
#     """
#     b64_image = _page_to_base64(pdf_path, page_number)

#     vision_llm = ChatGroq(
#         model=VISION_MODEL,
#         temperature=0.2,
#         groq_api_key=GROQ_API_KEY,
#     )

#     prompt = """Extract all information from this document page.

#     - Transcribe all readable text exactly.
#     - If there's a flowchart, diagram, or process flow: describe each step/box
#       in order, and describe the arrows/connections between them (e.g.
#       "Box A leads to Box B, which branches into Box C and Box D").
#     - If there's a chart, table, or graph: describe what it shows.
#     - Be thorough and factual — do not skip visual details."""

#     message = HumanMessage(
#         content=[
#             {"type": "text", "text": prompt},
#             {
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/png;base64,{b64_image}"},
#             },
#         ]
#     )

#     response = vision_llm.invoke([message])
#     return response.content.strip()


# def extract_documents_with_vision(path, min_text_len=40):
#     """
#     Extract text per page. If a page has little native text (scanned page,
#     or a page that's mostly a diagram/flowchart), fall back to a vision LLM
#     that reads and describes the page's visual content.
#     """
#     loader = PyPDFLoader(path)
#     pages = loader.load()

#     docs = []
#     for i, page_doc in enumerate(pages):
#         text = page_doc.page_content or ""

#         if len(text.strip()) < min_text_len:
#             try:
#                 vision_text = _describe_page_with_vision(path, i)
#                 text = f"{text}\n{vision_text}".strip()
#             except Exception as e:
#                 print(f"⚠️ Vision extraction failed on page {i}: {e}")

#         if text.strip():
#             docs.append(
#                 Document(
#                     page_content=text,
#                     metadata={"source": os.path.basename(path), "page": i},
#                 )
#             )

#     return docs


# # -----------------------------
# # 📚 PDF retriever setup
# # -----------------------------
# def create_pdf_retriever(file_paths):
#     all_docs = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunk_id = 0

#     for path in file_paths:
#         docs = extract_documents_with_vision(path)
#         chunks = text_splitter.split_documents(docs)
#         for chunk in chunks:
#             chunk.metadata = chunk.metadata or {}
#             chunk.metadata["chunk"] = chunk_id
#             chunk_id += 1
#         all_docs.extend(chunks)

#     if not all_docs:
#         raise ValueError(
#             "❌ No extractable text found in the uploaded PDF(s), even after vision analysis. "
#             "The file may be corrupted or contain no readable content."
#         )

#     vectorstore = FAISS.from_documents(all_docs, embeddings)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     return retriever


# # -----------------------------
# # 🌐 Serper API — fetch live web data (Google Search)
# # -----------------------------
# def fetch_live_data_from_serper(query):
#     url = "https://google.serper.dev/search"
#     headers = {
#         "X-API-KEY": SERPER_API_KEY,
#         "Content-Type": "application/json"
#     }
#     payload = {"q": query}
#     response = requests.post(url, headers=headers, json=payload)
#     data = response.json()

#     snippets = []
#     if "organic" in data:
#         for item in data["organic"]:
#             if "snippet" in item:
#                 snippets.append(item["snippet"])

#     if not snippets:
#         return "No relevant live data found."

#     return "\n".join(snippets[:5])  # use top 5 snippets


# # -----------------------------
# # 💬 Combine RAG + Serper + General fallback
# # -----------------------------
# def generate_answer(user_question, retriever, include_sources=True):
#     # Step 1: Retrieve document context
#     relevant_docs = retriever.invoke(user_question)
#     context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
#     sources = []
#     for doc in relevant_docs:
#         meta = doc.metadata or {}
#         sources.append({
#             "source": meta.get("source", "Unknown"),
#             "page": meta.get("page", 0),
#         })

#     # Step 2: Initialize Groq LLM
#     llm = ChatGroq(
#         model=TEXT_MODEL,
#         temperature=0.3,
#         groq_api_key=GROQ_API_KEY
#     )

#     # Step 3: If context found, use it
#     if context.strip():
#         prompt = f"""
#             You are a professional assistant helping answer questions using document context.
#             Write a clear, well-organized answer in your own words — do not copy raw text or
#             bullet-dump the context verbatim. Synthesize the information into a natural,
#             readable response, using short paragraphs or clean bullet points only where it
#             genuinely helps clarity.

#             Format the answer in Markdown: **bold** key terms, names, numbers, and important
#             facts. Use bullet points or numbered lists when listing multiple items (e.g. skills,
#             dates, steps). Use short sub-headings (##) if the answer covers more than one topic.

#             If the answer is not present in the context, respond only with: "Not found in PDFs."

#             Context:
#             {context}

#             Question:
#             {user_question}
#             """
#         response = llm.invoke(prompt)
#         answer_text = response.content

#         # Step 4: Fallback to Serper if PDF doesn't help
#         if "Not found in PDFs" in answer_text:
#             print("🔍 Falling back to live Serper search...")
#             answer_text = fetch_live_answer_with_serper(user_question, llm)
#             sources = []
#     else:
#         # Step 5: If no PDF context, go directly to Serper
#         print("🔍 No PDF context found — using live Serper search...")
#         answer_text = fetch_live_answer_with_serper(user_question, llm)
#         sources = []

#     # Step 6: Format sources for UI if requested
#     if include_sources:
#         if sources:
#             # Deduplicate by (source, page) so the same page isn't listed 4x for 4 chunks
#             seen = set()
#             unique_sources = []
#             for s in sources:
#                 key = (s["source"], s["page"])
#                 if key not in seen:
#                     seen.add(key)
#                     unique_sources.append(s)

#             source_lines = "\n".join([
#                 f"- {s['source']} (page {s['page'] + 1})" for s in unique_sources
#             ])
#             source_md = f"\n\n---\n### Sources\n{source_lines}"
#         else:
#             source_md = "\n\n---\n### Sources\nGenerated by LLM (no PDF match found)"

#         answer_text = f"{answer_text}{source_md}"

#     return answer_text


# # Use Serper API + LLM for live answers
# def fetch_live_answer_with_serper(question, llm):
#     serper_context = fetch_live_data_from_serper(question)

#     prompt = f"""
#     You are a factual assistant. Using the following live web search results,
#     extract the **most accurate and up-to-date information** related to the question.

#     Context:
#     {serper_context}

#     Question:
#     {question}

#     Only return the factual answer, not explanation.
#     """
#     response = llm.invoke(prompt)
#     return response.content.strip()


# # ================================================================
# # 7️⃣ Example usage
# # ================================================================
# # if __name__ == "__main__":
# #     pdf_files = ["data/sample1.pdf"]  # Change to your actual PDF paths
# #     retriever = create_pdf_retriever(pdf_files)

# #     print("✅ System ready! You can now ask live + RAG questions.")
# #     while True:
# #         query = input("\n🧠 Ask me anything: ")
# #         reply = generate_answer(query, retriever)
# #         print("\n🤖", reply)







# import os
# import requests
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq

# # 1️⃣ Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError("❌ GROQ_API_KEY not found in .env file.")
# if not SERPER_API_KEY:
#     raise ValueError("❌ SERPER_API_KEY not found in .env file.")

# # 2️⃣ Initialize Embedding model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # #  PDF retriever setup
# # def create_pdf_retriever(file_paths):
# #     all_docs = []
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# #     for path in file_paths:
# #         loader = PyPDFLoader(path)
# #         docs = loader.load()
# #         chunks = text_splitter.split_documents(docs)
# #         all_docs.extend(chunks)

# #     vectorstore = FAISS.from_documents(all_docs, embeddings)
# #     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# #     return retriever

# def create_pdf_retriever(file_paths):
#     all_docs = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunk_id = 0

#     for path in file_paths:
#         loader = PyPDFLoader(path)
#         docs = loader.load()
#         # Assume each doc has metadata (add if not)
#         for doc in docs:
#             doc.metadata = doc.metadata or {}
#             doc.metadata["source"] = os.path.basename(path)
#         chunks = text_splitter.split_documents(docs)
#         # Add chunk index and page number if available
#         for chunk in chunks:
#             chunk.metadata = chunk.metadata or {}
#             chunk.metadata["chunk"] = chunk_id
#             chunk_id += 1
#         all_docs.extend(chunks)

#     vectorstore = FAISS.from_documents(all_docs, embeddings)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     return retriever


# # Serper API — fetch live web data (Google Search)
# def fetch_live_data_from_serper(query):
#     url = "https://google.serper.dev/search"
#     headers = {
#         "X-API-KEY": SERPER_API_KEY,
#         "Content-Type": "application/json"
#     }
#     payload = {"q": query}
#     response = requests.post(url, headers=headers, json=payload)
#     data = response.json()

#     snippets = []
#     if "organic" in data:
#         for item in data["organic"]:
#             if "snippet" in item:
#                 snippets.append(item["snippet"])

#     if not snippets:
#         return "No relevant live data found."

#     return "\n".join(snippets[:5])  # use top 5 snippets


# # Combine RAG + Serper + General fallback
# # def generate_answer(user_question, retriever,):
# #     # Step 1: Retrieve document context
# #     relevant_docs = retriever.invoke(user_question)
# #     context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

# #     # Step 2: Initialize Groq LLM
# #     llm = ChatGroq(
# #         model="llama-3.1-8b-instant",
# #         temperature=0.3,
# #         groq_api_key=GROQ_API_KEY
# #     )

# #     # Step 3: If context found, use it
# #     if context.strip():
# #         prompt = f"""
# #         You are a helpful assistant. Use the following context to answer the question.
# #         If the answer isn't in the PDFs, just say: "Not found in PDFs."

# #         Context:
# #         {context}

# #         Question:
# #         {user_question}
# #         """
# #         response = llm.invoke(prompt)
# #         answer = response.content

# #         # Step 4: Fallback to Serper if PDF doesn't help
# #         if "Not found in PDFs" in answer:
# #             print("🔍 Falling back to live Serper search...")
# #             answer = fetch_live_answer_with_serper(user_question, llm)
# #     else:
# #         # Step 5: If no PDF context, go directly to Serper
# #         print("🔍 No PDF context found — using live Serper search...")
# #         answer = fetch_live_answer_with_serper(user_question, llm)

# #     return answer



# # def generate_answer(user_question, retriever, include_sources=True):
# #     # Step 1: Retrieve document context
# #     relevant_docs = retriever.invoke(user_question)
# #     context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
# #     sources = []
# #     for doc in relevant_docs:
# #         # You can add scoring info if your retriever provides it (e.g., similarity)
# #         meta = doc.metadata or {}
# #         sources.append({
# #             "score": meta.get("score", ""),  # If available
# #             "source": meta.get("source", ""),
# #             "page": meta.get("page", ""),
# #             "chunk": meta.get("chunk", ""),
# #         })

# #     # Step 2: Initialize Groq LLM
# #     llm = ChatGroq(
# #         model="llama-3.1-8b-instant",
# #         temperature=0.3,
# #         groq_api_key=GROQ_API_KEY
# #     )

# #     # Step 3: If context found, use it
# #     if context.strip():
# #         prompt = f"""
# #         You are a helpful assistant. Use the following context to answer the question.
# #         If the answer isn't in the PDFs, just say: "Not found in PDFs."

# #         Context:
# #         {context}

# #         Question:
# #         {user_question}
# #         """
# #         response = llm.invoke(prompt)
# #         answer_text = response.content

# #         # Step 4: Fallback to Serper if PDF doesn't help
# #         if "Not found in PDFs" in answer_text:
# #             print("🔍 Falling back to live Serper search...")
# #             answer_text = fetch_live_answer_with_serper(user_question, llm)
# #             sources = []
# #     else:
# #         # Step 5: If no PDF context, go directly to Serper
# #         print("🔍 No PDF context found — using live Serper search...")
# #         answer_text = fetch_live_answer_with_serper(user_question, llm)
# #         sources = []

# #     # Step 6: Format sources for UI if requested
# #     source_md = ""
# #     if include_sources:
# #         if sources:
# #             source_md = (
# #                 "\n**Sources**\n"
# #                 "| Score | Source | Page | Chunk |\n"
# #                 "|-------|--------|------|-------|\n" +
# #                 "\n".join([
# #                     f"| {item['score']} | {item['source']} | {item['page']} | {item['chunk']} |"
# #                     for item in sources
# #                 ])
# #             )
# #         else:
# #             source_md = (
# #                 "\n**Sources**\n"
# #                 "| Source |\n"
# #                 "|--------|\n"
# #                 "| Answer generated by LLM (No PDF source found) |"
# #             )

# #         answer_text = f"{answer_text}\n\n{source_md}"

# #     return answer_text


# def generate_answer(user_question, retriever, include_sources=True):
#     # Step 1: Retrieve document context
#     relevant_docs = retriever.invoke(user_question)
#     context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
#     sources = []
#     for doc in relevant_docs:
#         meta = doc.metadata or {}
#         sources.append({
#             "source": meta.get("source", "Unknown"),
#             "page": meta.get("page", 0),
#         })

#     # Step 2: Initialize Groq LLM
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant",
#         temperature=0.3,
#         groq_api_key=GROQ_API_KEY
#     )

#     # Step 3: If context found, use it
#     if context.strip():
#         prompt = f"""
#             You are a professional assistant helping answer questions using document context.
#             Write a clear, well-organized answer in your own words — do not copy raw text or
#             bullet-dump the context verbatim. Synthesize the information into a natural,
#             readable response, using short paragraphs or clean bullet points only where it
#             genuinely helps clarity.

#             Format the answer in Markdown: **bold** key terms, names, numbers, and important
#             facts. Use bullet points or numbered lists when listing multiple items (e.g. skills,
#             dates, steps). Use short sub-headings (##) if the answer covers more than one topic.

#             If the answer is not present in the context, respond only with: "Not found in PDFs."

#             Context:
#             {context}

#             Question:
#             {user_question}
#             """
#         response = llm.invoke(prompt)
#         answer_text = response.content

#         # Step 4: Fallback to Serper if PDF doesn't help
#         if "Not found in PDFs" in answer_text:
#             print("🔍 Falling back to live Serper search...")
#             answer_text = fetch_live_answer_with_serper(user_question, llm)
#             sources = []
#     else:
#         # Step 5: If no PDF context, go directly to Serper
#         print("🔍 No PDF context found — using live Serper search...")
#         answer_text = fetch_live_answer_with_serper(user_question, llm)
#         sources = []

#     # Step 6: Format sources for UI if requested
#     if include_sources:
#         if sources:
#             # Deduplicate by (source, page) so the same page isn't listed 4x for 4 chunks
#             seen = set()
#             unique_sources = []
#             for s in sources:
#                 key = (s["source"], s["page"])
#                 if key not in seen:
#                     seen.add(key)
#                     unique_sources.append(s)

#             source_lines = "\n".join([
#                 f"- {s['source']} (page {s['page'] + 1})" for s in unique_sources
#             ])
#             source_md = f"\n\n---\n**Sources**\n{source_lines}"
#         else:
#             source_md = "\n\n---\n**Sources**\nGenerated by LLM (no PDF match found)"

#         answer_text = f"{answer_text}{source_md}"

#     return answer_text

# # Use Serper API + LLM for live answers
# def fetch_live_answer_with_serper(question, llm):
#     serper_context = fetch_live_data_from_serper(question)

#     prompt = f"""
#     You are a factual assistant. Using the following live web search results,
#     extract the **most accurate and up-to-date information** related to the question.

#     Context:
#     {serper_context}

#     Question:
#     {question}

#     Only return the factual answer, not explanation.
#     """
#     response = llm.invoke(prompt)
#     return response.content.strip()


# # # ================================================================
# # # 7️⃣ Example usage
# # # ================================================================
# # if __name__ == "__main__":
# #     pdf_files = ["data/sample1.pdf"]  # Change to your actual PDF paths
# #     retriever = create_pdf_retriever(pdf_files)

# #     print("✅ System ready! You can now ask live + RAG questions.")
# #     while True:
# #         query = input("\n🧠 Ask me anything: ")
# #         reply = generate_answer(query, retriever)
# #         print("\n🤖", reply)
