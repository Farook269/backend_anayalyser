from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str


class HealthResponse(BaseModel):
    status: str


class UploadResponse(BaseModel):
    status: str
    file_uploaded: str

