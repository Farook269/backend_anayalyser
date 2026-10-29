"""
Backward-compatibility bridge for legacy executions.
Re-exports FastAPI app instance from modular `app.main`.
"""
import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("fastapi_api:app", host=host, port=port, reload=True)
