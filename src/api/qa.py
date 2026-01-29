from fastapi import APIRouter, Request, HTTPException
import structlog
import time

# Internal imports
from src.core.config import get_settings
from src.models.schemas import QuestionRequest, AnswerResponse, SourceChunk
from src.services.embeddings import get_embedding_service
from src.services.vector_store import get_vector_store
from src.services.llm_service import get_llm_service
from src.services.metrics import metrics, Timer

# 1. Initialize Logger and Settings
logger = structlog.get_logger(__name__)
settings = get_settings()

# 2. Define Router (Defined at the top level to avoid import errors)
router = APIRouter(prefix="/qa", tags=["QA"])

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: Request, query: QuestionRequest):
    """
    Ask a question using Gemini 3 Flash reasoning over retrieved context.
    """
    metrics.record_query()
    
    with Timer() as total_timer:
        try:
            # 1. Embed question
            embed_service = get_embedding_service()
            q_emb = embed_service.embed_text(query.question)
            
            # 2. Search for relevant context
            vector_store = get_vector_store()
            with Timer() as ret_timer:
                results = vector_store.search(q_emb, top_k=query.top_k)
            metrics.record_retrieval(ret_timer.elapsed_ms)
            
            # 3. Generate answer with Gemini 3
            llm_service = get_llm_service()
            with Timer() as gen_timer:
                generation_result = llm_service.generate_answer(query.question, results)
            metrics.record_generation(gen_timer.elapsed_ms)

            # 4. Prepare source citations
            sources = []
            if query.include_sources and results:
                for res in results:
                    sources.append(SourceChunk(
                        chunk_id=res.chunk_id,
                        document_id=res.document_id,
                        document_name=res.metadata.get("filename", "Unknown"),
                        content=res.content,
                        similarity_score=res.similarity_score,
                        chunk_index=res.chunk_index
                    ))

            # 5. Return structured response
            return AnswerResponse(
                question=query.question,
                answer=generation_result.answer,
                sources=sources if query.include_sources else None,
                confidence_score=llm_service.calculate_confidence(results),
                retrieval_time_ms=round(ret_timer.elapsed_ms, 2),
                generation_time_ms=round(gen_timer.elapsed_ms, 2),
                total_time_ms=round((time.perf_counter() - total_timer.start) * 1000, 2),
                model_used=generation_result.model,
                tokens_used=generation_result.tokens_used
            )

        except Exception as e:
            logger.error("Question answering failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"QA Error: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy", "router": "qa"}