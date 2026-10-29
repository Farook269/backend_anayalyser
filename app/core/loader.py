import os
import base64
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from app.core.llm import get_vision_llm


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

