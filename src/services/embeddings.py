"""Embedding generation service using Google's Generative AI models."""
import hashlib
import time
from typing import Optional, List
import numpy as np
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from src.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Configure the Gemini SDK
genai.configure(api_key=settings.gemini_api_key)

class EmbeddingService:
    """Service for generating text embeddings using Gemini API."""
    
    def __init__(self, model: str = None):
        self.model = model or settings.embedding_model
        self._cache: dict[str, List[float]] = {}
        logger.info("Initialized Gemini EmbeddingService", model=self.model)

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20))
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text chunk."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        text = text.strip()
        cache_key = self._get_cache_key(text)
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Rate limit protection for single calls
            time.sleep(1.0) 
            
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            embedding = result['embedding']
            
            if use_cache:
                self._cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error("Gemini embedding failed", error=str(e))
            raise

    def embed_texts(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Generate embeddings for multiple texts with strict rate limiting."""
        if not texts:
            return []

        all_embeddings = []
        # Smaller batch size for free tier
        batch_size = 10 
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Add a "breathing room" pause between batches
                time.sleep(2.0)
                
                result = genai.embed_content(
                    model=self.model,
                    content=batch,
                    task_type=task_type
                )
                all_embeddings.extend(result['embedding'])
                
            except Exception as e:
                logger.error("Batch failed", batch=i, error=str(e))
                # If hit rate limit, wait longer and raise error to let retry handle it
                if "429" in str(e):
                    time.sleep(10)
                raise
                
        return all_embeddings

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Singleton instance
_embedding_service: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service