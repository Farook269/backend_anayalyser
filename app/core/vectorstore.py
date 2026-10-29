from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.core.loader import extract_documents_with_vision
from app.core.embeddings import get_embeddings


def create_pdf_retriever(file_paths):
    """
    Parses PDF documents, splits content into cohesive page-level chunks,
    and returns a FAISS vectorstore retriever.
    """
    all_docs = []
    # Generous chunk size ensures full department sections and multi-item tables stay intact
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=250)
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever
