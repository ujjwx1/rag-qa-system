"""Advanced text chunking service. 
Splits long documents into smaller, meaningful pieces while respecting boundaries.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import tiktoken
import structlog
from src.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

@dataclass
class ChunkResult:
    """Result of a chunking operation."""
    content: str
    start_char: int
    end_char: int
    token_count: int
    chunk_index: int
    metadata: dict

class SemanticChunker:
    """
    Groups sentences together until they reach a certain size, 
    ensuring we don't cut off thoughts mid-sentence.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        # We use the GPT-4 tokenizer for counting tokens as a safe standard
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        # Regex to find sentence endings (. ! ?) followed by space or newline
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+'
        return [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[ChunkResult]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk_sentences = []
        current_token_count = 0
        char_offset = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If the chunk is full, save it
            if current_token_count + sentence_tokens > self.chunk_size:
                if current_chunk_sentences:
                    content = " ".join(current_chunk_sentences)
                    chunks.append(ChunkResult(
                        content=content,
                        start_char=char_offset,
                        end_char=char_offset + len(content),
                        token_count=current_token_count,
                        chunk_index=len(chunks),
                        metadata=metadata.copy()
                    ))
                    char_offset += len(content) + 1
                    
                    # Manage overlap (keep a few sentences for context)
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk_sentences):
                        s_tokens = self._count_tokens(s)
                        if overlap_tokens + s_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    current_chunk_sentences = overlap_sentences
                    current_token_count = overlap_tokens

            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Add the final piece
        if current_chunk_sentences:
            content = " ".join(current_chunk_sentences)
            chunks.append(ChunkResult(
                content=content,
                start_char=char_offset,
                end_char=char_offset + len(content),
                token_count=current_token_count,
                chunk_index=len(chunks),
                metadata=metadata.copy()
            ))

        return chunks

def chunk_text(text: str, metadata: Optional[dict] = None) -> List[ChunkResult]:
    chunker = SemanticChunker()
    return chunker.chunk(text, metadata)