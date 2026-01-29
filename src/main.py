"""FastAPI application entry point for the Gemini 3 RAG System."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from src.core.config import get_settings
from src.api.documents import router as documents_router
from src.api.qa import router as qa_router

# 1. Setup Logging
structlog.configure(
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)
settings = get_settings()

# 2. Lifespan Handler (Startup/Shutdown logic)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure folders exist
    logger.info("Starting Gemini 3 RAG System", env=settings.app_env)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: Cleanup if necessary
    logger.info("Shutting down system")

# 3. Initialize FastAPI
app = FastAPI(
    title="Gemini 3 RAG QA System",
    description="A high-performance RAG system powered by Gemini 3 Flash Preview.",
    version="1.0.0",
    lifespan=lifespan
)

# 4. Add CORS (Allows you to connect a frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Include Routers
app.include_router(documents_router)
app.include_router(qa_router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "model": settings.llm_model,
        "docs": "/docs"
    }