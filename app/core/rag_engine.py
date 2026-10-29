from app.core.llm import llm


def generate_answer(user_question, retriever=None, include_sources=True):
    """
    Universal, domain-agnostic Document Intelligence RAG Pipeline.
    Adapts dynamically to any document type with structured formatting,
    bold metrics, and clean grouped source provenance.
    """
    # Case A: Document is loaded in memory -> Vector RAG + Strict Grounding
    if retriever is not None:
        # Collect all active document names dynamically
        doc_names = []
        if hasattr(retriever, "vectorstore") and hasattr(retriever.vectorstore, "docstore"):
            try:
                for doc in retriever.vectorstore.docstore._dict.values():
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        src = doc.metadata["source"]
                        if src not in doc_names:
                            doc_names.append(src)
            except Exception:
                pass

        doc_name = ", ".join(doc_names) if doc_names else "Uploaded Document"

        q_lower = user_question.lower().strip()
        is_greeting = any(
            q_lower == g or q_lower.startswith(g + " ")
            for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        ) and len(q_lower.split()) <= 4

        if is_greeting:
            return f"Hello! I have indexed **{doc_name}** and am ready to assist you. What would you like to explore?"

        is_overview_request = any(
            w in q_lower
            for w in [
                "summary", "summarize", "overview", "what is this document",
                "what is this pdf", "about this pdf", "explain this file",
                "what are these documents", "overview of these files"
            ]
        )
        retrieval_query = f"{user_question} overview key points summary" if is_overview_request else user_question

        # Strict document grounding & vector distance thresholding
        relevant_docs = []
        if hasattr(retriever, "vectorstore"):
            results_with_scores = retriever.vectorstore.similarity_search_with_score(retrieval_query, k=5)
            best_score = results_with_scores[0][1] if results_with_scores else 999.0
            if not is_overview_request and best_score > 1.15:
                return f"The requested information is not mentioned in the uploaded document ({doc_name})."

            # Filter out distant chunks to keep response focused on the user's specific query
            if is_overview_request:
                relevant_docs = [doc for doc, score in results_with_scores]
            else:
                max_score_cutoff = max(0.65, best_score + 0.18)
                relevant_docs = [
                    doc for doc, score in results_with_scores
                    if score <= max_score_cutoff and score < 0.88
                ]
                if not relevant_docs and results_with_scores:
                    relevant_docs = [results_with_scores[0][0]]
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

        # Universal prompt with executive visual structuring and mandatory metric bolding
        prompt = f"""You are an enterprise AI Document Intelligence Analyst assisting with: {doc_name}.

Operational Directives:
1. Strict Grounding: Answer strictly using ONLY the provided Document Context below. Never speculate, guess, or use outside knowledge.
2. Query Relevancy & Scope:
   - Answer specifically and directly what the user is asking for.
   - Do not dump unrelated sections, chapters, or clauses that the user did not ask about.
3. Visual Structure & Bolding (Essential):
   - When an inquiry covers multiple items, policies, entitlements, or rules, ALWAYS format each item as a distinct markdown bullet point (`- **Item Name:** ...`). Never merge multiple distinct facts into a single unbroken paragraph.
   - For sequential procedures or workflows, use numbered markdown steps (`1. **Step Name**: ...`).
   - ALWAYS format all key numbers, figures, metrics, quantities, deadlines, dates, and amounts in **bold** (e.g. **22 days**, **12 days**, **6 days**, **8 calendar days**).
4. Missing Information: If any requested fact is absent from the Document Context, state clearly:
   "The requested information is not mentioned in the uploaded document."

Document Context:
{context}

User Query: {user_question}
"""

        response = llm.invoke(prompt)
        clean_answer = response.content.strip()

        # Attach Verified Source Citations ONLY when information is actually found in the context
        not_found_phrases = [
            "not mentioned", "not found", "does not mention",
            "is not present", "cannot be found", "no mention"
        ]
        is_not_found = any(p in clean_answer.lower() for p in not_found_phrases)

        if include_sources and not is_not_found and sources:
            # Group pages by document and sort chronologically (e.g. Pages 1, 2, 3)
            doc_to_pages = {}
            for s in sources:
                d_name = s.get("source", "Document")
                p_num = s.get("page", 0) + 1
                if d_name not in doc_to_pages:
                    doc_to_pages[d_name] = set()
                doc_to_pages[d_name].add(p_num)

            source_lines = []
            for d_name, p_set in doc_to_pages.items():
                sorted_pages = sorted(list(p_set))
                if len(sorted_pages) == 1:
                    pages_label = f"Page {sorted_pages[0]}"
                else:
                    pages_label = f"Pages {', '.join(map(str, sorted_pages))}"
                source_lines.append(f"- **{d_name}** ({pages_label})")

            source_block = "\n".join(source_lines)
            clean_answer = f"{clean_answer}\n\n---\n**Verified Sources:**\n{source_block}"

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
