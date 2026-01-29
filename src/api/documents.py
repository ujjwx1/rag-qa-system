from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from src.models.database import db, Document, DocumentStatus
from src.services.background_tasks import queue_document_processing
import uuid, shutil, os

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    file_path = f"uploads/{doc_id}_{file.filename}"
    
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    new_doc = Document(id=doc_id, filename=file.filename, file_path=file_path, file_type=file.content_type, file_size=0)
    db.add_document(new_doc)
    
    # Trigger background processing
    queue_document_processing(doc_id)
    
    return {"document_id": doc_id, "status": "queued", "message": "Processing started in background."}

@router.get("/{doc_id}")
async def get_status(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc: raise HTTPException(status_code=404, detail="Not found")
    return doc