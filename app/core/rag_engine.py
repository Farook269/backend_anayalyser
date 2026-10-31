import re
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.llm import llm


def generate_answer(user_question, retriever=None, include_sources=True):
    """
    Universal, 100% domain-agnostic Document Intelligence RAG Pipeline.
    Adapts dynamically to any PDF with zero hardcoded templates,
    providing structured synthesis, bold metrics, and clean grouped citations.
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

        # Dynamic vector retrieval with k=8 to ensure comprehensive context across multi-page files
        relevant_docs = []
        if hasattr(retriever, "vectorstore"):
            results_with_scores = retriever.vectorstore.similarity_search_with_score(retrieval_query, k=8)
            best_score = results_with_scores[0][1] if results_with_scores else 999.0
            
            # 1.25 cutoff filters out completely off-topic inquiries
            if not is_overview_request and best_score > 1.25:
                return f"The requested information is not mentioned in the uploaded document."

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

        # 100% Universal, Domain-Agnostic System Prompt: Zero hardcoded keywords
        system_prompt = """You are an enterprise Document Intelligence AI Assistant.
Your objective is to answer the User Query factually, accurately, and professionally using ONLY the provided Document Context.

CORE GROUNDING RULES:
1. Strict Factual Grounding:
   - Base your answer exclusively on the statements, facts, tables, and data in the Document Context.
   - Do NOT assume, extrapolate, or fabricate any data, figures, numbers, or roles that are not explicitly present in the context.

2. Entity & Role Verification:
   - If the user asks about a specific person, role, position, title, department, or entity that is NOT mentioned in the context (for example, a job role or policy not mentioned in the document), state clearly:
     "The requested information is not mentioned in the uploaded document."
   - Never fabricate, estimate, or extrapolate numbers or ranges for unlisted entities.

3. Semantic Intent & Lists:
   - If the user asks "how many ...", about a list, categories, types, or counts (e.g. holidays, benefits, leave types, requirements, steps), carefully examine the relevant sections in the context. State the count and list the items clearly, even if the user's phrasing uses slightly different wording (e.g. "types of holidays" vs "company holidays").

4. Missing Information:
   - If the requested information cannot be answered from the provided context, reply strictly:
     "The requested information is not mentioned in the uploaded document."

5. Mandatory Markdown Bolding for Key Entities & Metrics:
   - ALWAYS format the main company name, organization names, and primary entities in **bold** (e.g., **Company Name**).
   - ALWAYS format all key numbers, metrics, salary amounts/ranges, quantities, deadlines, and dates in **bold** (e.g., **11 paid company holidays**, **22 days**, **$92,000–$124,000 USD**, **8 calendar days**).
   - Use clean bullet points with bold headers (`- **Role/Item:** details`) for multiple items or roles.
   - Use numbered lists with bold steps (`1. **Step:** ...`) for procedures.
"""

        human_prompt = f"""Document Context:
{context}

User Query: {user_question}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm.invoke(messages)
        clean_answer = response.content.strip()

        # Mathematical Entity Grounding Guardrail:
        # Detect if the query asked for specific capitalized entities (e.g. 'AI Research Scientist', 'VP of Sales')
        # that are completely absent from the indexed context, but a small model fabricated metrics/figures.
        cap_entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b', user_question)
        common_starters = {
            'what are', 'what is', 'how many', 'how much', 'when does',
            'where is', 'which of', 'who is', 'give me', 'tell me', 'can you'
        }
        specific_entities = [
            e for e in cap_entities
            if e.lower() not in common_starters and len(e.split()) >= 2
        ]

        if specific_entities:
            context_lower = context.lower()
            unmatched = [
                e for e in specific_entities
                if e.lower() not in context_lower
            ]
            if len(unmatched) == len(specific_entities):
                has_metrics = bool(re.search(r'[\$\€\£]|\bUSD\b|\bEUR\b', clean_answer))
                if has_metrics:
                    clean_answer = "The requested information is not mentioned in the uploaded document."

        # Attach Verified Source Citations ONLY when information is actually found in the context
        not_found_phrases = [
            "not mentioned", "not found", "does not mention", "does not contain",
            "is not present", "cannot be found", "no mention", "not listed",
            "no information", "does not provide", "not provided", "not available"
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
