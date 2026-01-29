"""Vector store service using FAISS for local similarity search."""
import os
import pickle
from pathlib import Path
from typing import Optional, List
import numpy as np
import faiss
import structlog
from src.core.config import get_settings
from src.models.database import db

logger = structlog.get_logger(__name__)
settings = get_settings()

class SearchResult:
    """Result from a similarity search."""
    def __init__(self, chunk_id, document_id, content, similarity_score, chunk_index, metadata):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.similarity_score = similarity_score
        self.chunk_index = chunk_index
        self.metadata = metadata

class FAISSVectorStore:
    """FAISS-based vector store for high-speed local search."""
    
    def __init__(self, dimension: int = 768):
        # Note: Gemini's embedding-001 uses 768 dimensions. 
        # If using text-embedding-004, it might be 768 or 1536.
        self.dimension = dimension
        self.index_path = settings.faiss_index_path
        
        # Mapping helpers
        self.idx_to_id = {}
        self._initialize_index()
        logger.info("Initialized FAISS vector store", dimension=self.dimension)

    def _initialize_index(self):
        """Load an existing index from disk or create a new one."""
        os.makedirs(self.index_path, exist_ok=True)
        index_file = Path(self.index_path) / "index.faiss"
        mapping_file = Path(self.index_path) / "mappings.pkl"

        if index_file.exists() and mapping_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(mapping_file, 'rb') as f:
                    self.idx_to_id = pickle.load(f)
                logger.info("Loaded existing FAISS index", total=self.index.ntotal)
            except Exception as e:
                logger.warning("Failed to load index, starting fresh", error=str(e))
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        # IndexFlatIP uses Inner Product (ideal for Cosine Similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.idx_to_id = {}

    def add_embeddings(self, ids: List[str], embeddings: List[List[float]]):
        """Add vectors to the index and save to disk."""
        if not ids or not embeddings:
            return

        # Convert to float32 NumPy array (FAISS requirement)
        vectors = np.array(embeddings).astype('float32')
        
        # L2 Normalize for Cosine Similarity
        faiss.normalize_L2(vectors)
        
        start_idx = self.index.ntotal
        self.index.add(vectors)

        # Map internal FAISS numeric IDs back to our string IDs (e.g., chunk_doc123_0)
        for i, doc_id in enumerate(ids):
            self.idx_to_id[start_idx + i] = doc_id
        
        self._save_to_disk()
        logger.info("Added vectors to FAISS", count=len(ids), total=self.index.ntotal)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for the most similar chunks."""
        if self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vec)

        # Search
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            
            chunk_id = self.idx_to_id.get(idx)
            chunk = db.get_chunk(chunk_id)
            
            if chunk:
                # Normalize similarity score to 0-1 range
                norm_score = float((score + 1) / 2)
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    similarity_score=norm_score,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata
                ))
        return results

    def _save_to_disk(self):
        faiss.write_index(self.index, str(Path(self.index_path) / "index.faiss"))
        with open(Path(self.index_path) / "mappings.pkl", 'wb') as f:
            pickle.dump(self.idx_to_id, f)

# Singleton
_vector_store = None
def get_vector_store() -> FAISSVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
    return _vector_store