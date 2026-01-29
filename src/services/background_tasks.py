"""Background task processing for document ingestion."""
import time
from concurrent.futures import ThreadPoolExecutor
import structlog
from src.models.database import db, DocumentStatus, Chunk # Imported Chunk model
from src.services.document_loader import load_document
from src.services.chunker import chunk_text
from src.services.embeddings import get_embedding_service
from src.services.vector_store import get_vector_store
from src.services.metrics import metrics

logger = structlog.get_logger(__name__)

# We allow 3 documents to be processed at the same time
executor = ThreadPoolExecutor(max_workers=3)

def process_document_task(document_id: str):
    """The actual heavy lifting: Load -> Chunk -> Embed -> Store"""
    start_time = time.time()
    try:
        doc = db.get_document(document_id)
        if not doc: return

        db.update_document(document_id, status=DocumentStatus.PROCESSING)
        
        # 1. Load
        text = load_document(doc.file_path)
        
        # 2. Chunk
        chunk_results = chunk_text(text, metadata={"filename": doc.filename})
        
        # 3. Embed
        embed_service = get_embedding_service()
        chunk_texts = [c.content for c in chunk_results]
        embeddings = embed_service.embed_texts(chunk_texts)
        
        # 4. Store
        vector_store = get_vector_store()
        chunk_ids = []
        
        # --- CRITICAL FIX HERE ---
        # We must convert ChunkResult (from chunker) into Chunk (database model)
        # so it has the 'document_id' field.
        for i, (chunk_res, emb) in enumerate(zip(chunk_results, embeddings)):
            c_id = f"chunk_{document_id}_{i}"
            
            # Create the database object
            db_chunk = Chunk(
                id=c_id,
                document_id=document_id,  # This was missing before!
                content=chunk_res.content,
                chunk_index=i,
                start_char=chunk_res.start_char,
                end_char=chunk_res.end_char,
                token_count=chunk_res.token_count,
                embedding=emb,
                metadata=chunk_res.metadata
            )
            
            # Save to in-memory DB
            db.chunks[c_id] = db_chunk 
            chunk_ids.append(c_id)
        # -------------------------
            
        vector_store.add_embeddings(chunk_ids, embeddings)
        
        # Finalize
        processing_time = time.time() - start_time
        db.update_document(
            document_id, 
            status=DocumentStatus.COMPLETED,
            chunks_count=len(chunk_results),
            processing_time_seconds=processing_time
        )
        # Now this method exists!
        metrics.record_document_processed(len(chunk_results))
        
        logger.info("Document processed successfully", doc_id=document_id)
        
    except Exception as e:
        logger.error("Background processing failed", doc_id=document_id, error=str(e))
        db.update_document(document_id, status=DocumentStatus.FAILED, error_message=str(e))

def queue_document_processing(document_id: str):
    """Adds the task to the worker pool."""
    executor.submit(process_document_task, document_id)
    logger.info("Document queued", doc_id=document_id)