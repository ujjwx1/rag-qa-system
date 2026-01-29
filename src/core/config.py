from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "RAG QA System"
    app_env: Literal["development", "production", "testing"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Gemini Configuration
    gemini_api_key: str
    # CRITICAL CHANGE: Switched to the newer, generous text-embedding-004
    embedding_model: str = "models/text-embedding-004"
    llm_model: str = "gemini-1.5-flash"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Rate Limiting
    rate_limit_per_minute: int = 30

    # Chunking Strategy
    chunk_size: int = 512
    chunk_overlap: int = 50

    # File Upload
    max_file_size_mb: int = 10
    allowed_extensions: list[str] = ["pdf", "txt", "png"]
    upload_dir: str = "uploads"

    # Vector Store
    vector_store_type: Literal["faiss", "pinecone"] = "faiss"
    faiss_index_path: str = "faiss_index"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

@lru_cache
def get_settings() -> Settings:
    return Settings()