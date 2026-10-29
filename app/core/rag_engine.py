from app.core.llm import llm


def generate_answer(user_question, retriever=None, include_sources=True):
    """
    Enterprise RAG pipeline with strict document grounding and zero hallucination.
    """
    # Case A: Document is loaded in memory -> Vector RAG + Strict Grounding
    if retriever is not None:
        doc_name = "Uploaded Document"
        try:
            sample_docs = retriever.invoke("document overview")
            if sample_docs and sample_docs[0].metadata:
                doc_name = sample_docs[0].metadata.get("source", doc_name)
        except Exception:
            pass

        q_lower = user_question.lower().strip()
        is_greeting = any(
            q_lower == g or q_lower.startswith(g + " ")
            for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        ) and len(q_lower.split()) <= 4

        if is_greeting:
            return f"Hello! I have indexed **{doc_name}** and am ready to assist you. What would you like to explore from this document?"

        is_overview_request = any(
            w in q_lower
            for w in [
                "summary", "summarize", "overview", "what is this document",
                "what is this pdf", "about this pdf", "explain this file"
            ]
        )
        retrieval_query = f"{user_question} overview key points summary" if is_overview_request else user_question

        # Strict document grounding: check vector distance to immediately prevent hallucinating outside facts
        relevant_docs = []
        if hasattr(retriever, "vectorstore"):
            results_with_scores = retriever.vectorstore.similarity_search_with_score(retrieval_query, k=5)
            best_score = results_with_scores[0][1] if results_with_scores else 999.0
            if not is_overview_request and best_score > 1.15:
                return f"The requested information is not mentioned in the uploaded document ('{doc_name}')."
            relevant_docs = [doc for doc, score in results_with_scores]
        else:
            relevant_docs = retriever.invoke(retrieval_query)

        context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

        sources = []
        for doc in relevant_docs:
            meta = doc.metadata or {}
            sources.append({
                "source": meta.get("source", doc_name),
                "page": meta.get("page", 0),
            })

        prompt = f"""You are an executive AI Document Intelligence Analyst assisting with the document '{doc_name}'.

Operational Directives:
1. Strict Grounding: Use ONLY facts directly stated in the Document Context below.
2. Professional Tone & Full Sentences:
   - Provide a complete, articulate, professional answer tailored to the user's inquiry.
   - For entity or specific questions (such as a name, title, contact, role, or date), provide a polished, complete sentence rather than just a raw word or fragment (e.g. "The candidate identified in this document is **Mulla Farook Basha**.").
   - Highlight key names, job titles, technologies, metrics, and dates in **bold**.
3. Structured Presentation: When presenting skills, career history, or multiple points, format them with clean bullet points.
4. Missing Information: If the requested information is absent from the Document Context, reply strictly with:
   "The requested information is not mentioned in the uploaded document ('{doc_name}')."

Document Context ('{doc_name}'):
{context}

User Query: {user_question}
"""

        response = llm.invoke(prompt)
        clean_answer = response.content.strip()

        # Attach Verified Source Citations when answering from document context
        is_not_found = "not mentioned in the uploaded document" in clean_answer.lower()
        if include_sources and not is_not_found and sources:
            seen = set()
            unique_sources = []
            for s in sources:
                key = (s["source"], s["page"])
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)

            source_lines = "\n".join([
                f"- **{s['source']}** (Page {s['page'] + 1})" for s in unique_sources
            ])
            clean_answer = f"{clean_answer}\n\n---\n**Verified Sources:**\n{source_lines}"

        return clean_answer

    # Case B: No document uploaded yet -> Direct user to upload document
    else:
        q_lower = user_question.lower().strip()
        is_greeting = any(
            q_lower == g or q_lower.startswith(g + " ")
            for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        ) and len(q_lower.split()) <= 4
        if is_greeting:
            return "Hello! Please upload a PDF document in the sidebar to begin analysis."

        return "Please upload a PDF document in the sidebar to begin analysis. Once uploaded, I will analyze its contents and answer questions strictly based on the file."
