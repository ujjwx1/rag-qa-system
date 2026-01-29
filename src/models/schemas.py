"""Pydantic schemas for request/response validation."""
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# ==================== Document Schemas ====================
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunks_count: Optional[int] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    created_at: datetime
    updated_at: datetime

class DocumentListResponse(BaseModel):
    total_count: int
    documents: List[DocumentStatusResponse]

# ==================== Question/Answer Schemas ====================
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty whitespace")
        return v.strip()

class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    similarity_score: float
    chunk_index: int

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[SourceChunk]] = None
    confidence_score: float
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_used: str
    tokens_used: int

# ==================== Error & Metrics ====================
class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)