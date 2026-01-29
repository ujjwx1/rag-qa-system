"""In-memory database models for document and chunk management."""
import threading
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Document:
    """Represents an uploaded document."""
    id: str
    filename: str
    file_path: str
    file_type: str
    file_size: int
    status: DocumentStatus = DocumentStatus.PENDING
    chunks_count: int = 0
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class InMemoryDatabase:
    """Thread-safe in-memory database for documents and chunks."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.document_chunks: Dict[str, List[str]] = {}
        self._data_lock = threading.RLock()

    def add_document(self, document: Document) -> None:
        with self._data_lock:
            self.documents[document.id] = document
            self.document_chunks[document.id] = []

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._data_lock:
            return self.documents.get(doc_id)

    def update_document(self, doc_id: str, **kwargs) -> Optional[Document]:
        with self._data_lock:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                for key, value in kwargs.items():
                    if hasattr(doc, key):
                        setattr(doc, key, value)
                doc.updated_at = datetime.utcnow()
                return doc
            return None

    def list_documents(self) -> List[Document]:
        with self._data_lock:
            return list(self.documents.values())

    def add_chunks_batch(self, chunks: List[Chunk]) -> None:
        with self._data_lock:
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
                if chunk.document_id in self.document_chunks:
                    self.document_chunks[chunk.document_id].append(chunk.id)

    def get_chunks_count(self) -> int:
        with self._data_lock:
            return len(self.chunks)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._data_lock:
            return self.chunks.get(chunk_id)

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        with self._data_lock:
            chunk_ids = self.document_chunks.get(doc_id, [])
            return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

# Singleton instance
db = InMemoryDatabase()