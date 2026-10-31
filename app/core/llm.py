import sys
try:
    import langchain_core.pydantic_v1
except (ImportError, AttributeError):
    try:
        import pydantic.v1 as pydantic_v1
        import langchain_core
        sys.modules['langchain_core.pydantic_v1'] = pydantic_v1
    except Exception:
        pass

from app.config import USE_LOCAL_LLM, LOCAL_MODEL_NAME, OLLAMA_BASE_URL, GROQ_API_KEY

TEXT_MODEL = "qwen/qwen3.8-27b"
VISION_MODEL = "qwen/qwen3.8-27b"

# Primary Chat LLM Initialization
if USE_LOCAL_LLM:
    from langchain_community.chat_models import ChatOllama
    print(f"[INFO] Initializing 100% Local LLM via Ollama ({LOCAL_MODEL_NAME}) at {OLLAMA_BASE_URL}")
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LOCAL_MODEL_NAME,
        temperature=0.2,
        num_predict=400,
    )
else:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=TEXT_MODEL,
        temperature=0.3,
        max_tokens=600,
        groq_api_key=GROQ_API_KEY
    )

# Vision LLM (for scanned documents and complex diagrams)
_vision_llm = None


def get_vision_llm():
    global _vision_llm
    if _vision_llm is None:
        if USE_LOCAL_LLM:
            from langchain_community.chat_models import ChatOllama
            _vision_llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model="llama3.2-vision:latest",
                temperature=0.2,
            )
        else:
            from langchain_groq import ChatGroq
            _vision_llm = ChatGroq(
                model=VISION_MODEL,
                temperature=0.2,
                max_tokens=500,
                groq_api_key=GROQ_API_KEY,
            )
    return _vision_llm

