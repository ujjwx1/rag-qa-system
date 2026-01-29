"""Document loading and text extraction service."""
import os
from pathlib import Path
from typing import Protocol
import structlog
from PyPDF2 import PdfReader

logger = structlog.get_logger(__name__)

class DocumentLoader(Protocol):
    def load(self, file_path: str) -> str: ...
    def supported_extensions(self) -> list[str]: ...

class PDFLoader:
    """Extracts text from PDF files."""
    def supported_extensions(self) -> list[str]:
        return ["pdf"]

    def load(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error("Failed to load PDF", file_path=file_path, error=str(e))
            raise ValueError(f"Failed to process PDF: {str(e)}")

class TXTLoader:
    """Extracts text from TXT files."""
    def supported_extensions(self) -> list[str]:
        return ["txt"]

    def load(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

class DocumentLoaderFactory:
    """Directs the file to the correct loader based on extension."""
    _loaders = {}

    @classmethod
    def register_loader(cls, loader):
        for ext in loader.supported_extensions():
            cls._loaders[ext.lower()] = loader

    @classmethod
    def get_loader(cls, file_path: str):
        ext = Path(file_path).suffix.lower().lstrip('.')
        if ext not in cls._loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        return cls._loaders[ext]

# Register the loaders
DocumentLoaderFactory.register_loader(PDFLoader())
DocumentLoaderFactory.register_loader(TXTLoader())

def load_document(file_path: str) -> str:
    loader = DocumentLoaderFactory.get_loader(file_path)
    return loader.load(file_path)