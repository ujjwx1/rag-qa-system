"""LLM service for generating answers using Gemini 3 Flash Preview."""
from typing import Optional, List
from dataclasses import dataclass
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from src.core.config import get_settings
from src.services.vector_store import SearchResult

logger = structlog.get_logger(__name__)
settings = get_settings()

@dataclass
class GenerationResult:
    """Result from Gemini 3 generation."""
    answer: str
    tokens_used: int
    model: str
    finish_reason: str

class LLMService:
    """Service for generating answers using Gemini 3's advanced reasoning."""

    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.
Instructions:
1. Answer ONLY using the provided context chunks.
2. If the context doesn't have the answer, say "I couldn't find that in the documents."
3. Cite your sources by mentioning which Chunk # you are referring to.
4. Use a professional and concise tone.
"""

    def __init__(self, model: str = None):
        self.model = model or settings.llm_model
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        
        # FIX: Removed system_instruction from here to prevent version errors
        self.client = genai.GenerativeModel(model_name=self.model)
        
        logger.info("Initialized Gemini 3 LLM Service", model=self.model)

    def _build_context_prompt(self, question: str, search_results: List[SearchResult]) -> str:
        """Format the retrieved chunks into a single prompt string."""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Chunk {i}]\n{result.content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # FIX: We now include the SYSTEM_PROMPT here directly
        return f"{self.SYSTEM_PROMPT}\n\nContext from documents:\n{context_text}\n\nQuestion: {question}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_answer(self, question: str, search_results: List[SearchResult]) -> GenerationResult:
        if not search_results:
            return GenerationResult("No relevant context found.", 0, self.model, "no_context")

        # The prompt now contains the system instructions
        full_prompt = self._build_context_prompt(question, search_results)

        try:
            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    candidate_count=1,
                )
            )

            # Safe token access
            tokens = 0
            if hasattr(response, 'usage_metadata'):
                tokens = response.usage_metadata.total_token_count

            return GenerationResult(
                answer=response.text,
                tokens_used=tokens,
                model=self.model,
                finish_reason="STOP"
            )
        except Exception as e:
            logger.error("Gemini 3 generation failed", error=str(e))
            raise

    def calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """Simple confidence score based on similarity scores."""
        if not search_results:
            return 0.0
        scores = [r.similarity_score for r in search_results]
        return sum(scores[:3]) / len(scores[:3])

# Singleton instance
_llm_service = None
def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service