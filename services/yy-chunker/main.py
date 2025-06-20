import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.models import (
    ChunkingRequest, ChunkingResponse, ChunkingStatus,
    ChunkingMethod, ChunkingResult, ChunkMetadata
)
from api.chunkers import DocumentChunker
from api.config import settings

# Configure logging
logger.add(
    "logs/yy-chunker.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.LOG_LEVEL
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting YY-Chunker Service...")
    
    # Initialize document chunker
    app.state.chunker = DocumentChunker()
    await app.state.chunker.initialize()
    
    logger.info("YY-Chunker Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down YY-Chunker Service...")
    await app.state.chunker.cleanup()
    logger.info("YY-Chunker Service stopped")


# Create FastAPI application
app = FastAPI(
    title="DEAAP YY-Chunker",
    description="Advanced document chunking service with multiple strategies",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "yy-chunker",
        "version": "1.0.0",
        "supported_methods": [method.value for method in ChunkingMethod],
        "timestamp": str(asyncio.get_event_loop().time())
    }


@app.post("/chunk", response_model=ChunkingResponse)
async def chunk_document(
    content: str = Form(..., description="Document content to chunk"),
    method: ChunkingMethod = Form(default=ChunkingMethod.SEMANTIC),
    chunk_size: int = Form(default=2100, ge=100, le=8000),
    chunk_overlap: int = Form(default=200, ge=0),
    min_chunk_size: int = Form(default=100, ge=50),
    max_chunks: int = Form(default=1000, ge=1),
    language: str = Form(default="en"),
    preserve_structure: bool = Form(default=True)
):
    """Chunk document content using specified method"""
    try:
        # Create chunking request
        request = ChunkingRequest(
            content=content,
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            language=language,
            preserve_structure=preserve_structure
        )
        
        # Submit for chunking
        task_id = await app.state.chunker.submit_chunking(request)
        
        return ChunkingResponse(
            task_id=task_id,
            status=ChunkingStatus.QUEUED,
            message="Document chunking queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chunk-file", response_model=ChunkingResponse)
async def chunk_file(
    file: UploadFile = File(...),
    method: ChunkingMethod = Form(default=ChunkingMethod.SEMANTIC),
    chunk_size: int = Form(default=2100, ge=100, le=8000),
    chunk_overlap: int = Form(default=200, ge=0),
    min_chunk_size: int = Form(default=100, ge=50),
    max_chunks: int = Form(default=1000, ge=1),
    language: str = Form(default="en"),
    preserve_structure: bool = Form(default=True)
):
    """Chunk document file using specified method"""
    try:
        # Read file content
        content = await file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Create chunking request
        request = ChunkingRequest(
            content=content,
            filename=file.filename,
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            language=language,
            preserve_structure=preserve_structure
        )
        
        # Submit for chunking
        task_id = await app.state.chunker.submit_chunking(request)
        
        return ChunkingResponse(
            task_id=task_id,
            status=ChunkingStatus.QUEUED,
            message="Document chunking queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Error chunking file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}")
async def get_chunking_status(task_id: str):
    """Get chunking task status"""
    try:
        status = await app.state.chunker.get_task_status(task_id)
        return status
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=404, detail="Task not found")


@app.get("/result/{task_id}")
async def get_chunking_result(task_id: str):
    """Get chunking result"""
    try:
        result = await app.state.chunker.get_task_result(task_id)
        return result
    except Exception as e:
        logger.error(f"Error getting task result: {str(e)}")
        raise HTTPException(status_code=404, detail="Result not found")


@app.get("/methods")
async def list_chunking_methods():
    """List available chunking methods"""
    return {
        "methods": [
            {
                "name": method.value,
                "description": ChunkingMethod.get_description(method)
            }
            for method in ChunkingMethod
        ]
    }


@app.get("/tasks")
async def list_chunking_tasks(
    limit: int = 10,
    offset: int = 0,
    status: Optional[ChunkingStatus] = None
):
    """List chunking tasks"""
    try:
        tasks = await app.state.chunker.list_tasks(limit, offset, status)
        return tasks
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
