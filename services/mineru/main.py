import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import aiofiles

from api.models import (
    ProcessingRequest, ProcessingResponse, ProcessingStatus,
    DocumentType, OutputFormat, ExtractionResult
)
from api.processors import DocumentProcessor
from api.config import settings

# Configure logging
logger.add(
    "logs/mineru.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.LOG_LEVEL
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Mineru Engine Service...")
    
    # Initialize document processor
    app.state.processor = DocumentProcessor()
    await app.state.processor.initialize()
    
    logger.info("Mineru Engine Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Mineru Engine Service...")
    await app.state.processor.cleanup()
    logger.info("Mineru Engine Service stopped")


# Create FastAPI application
app = FastAPI(
    title="DEAAP Mineru Engine",
    description="PDF to Markdown conversion service using Magic-PDF and Mineru",
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
        "service": "mineru-engine",
        "version": "1.0.0",
        "timestamp": str(asyncio.get_event_loop().time())
    }


@app.post("/convert", response_model=ProcessingResponse)
async def convert_document(
    file: UploadFile = File(...),
    output_format: OutputFormat = Form(default=OutputFormat.MARKDOWN),
    extract_images: bool = Form(default=True),
    extract_tables: bool = Form(default=True),
    ocr_enabled: bool = Form(default=True),
    language: str = Form(default="en")
):
    """Convert PDF or other document formats to markdown"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.1f}MB"
            )
        
        # Reset file pointer
        await file.seek(0)
        
        # Create processing request
        request = ProcessingRequest(
            filename=file.filename,
            output_format=output_format,
            extract_images=extract_images,
            extract_tables=extract_tables,
            ocr_enabled=ocr_enabled,
            language=language
        )
        
        # Submit for processing
        task_id = await app.state.processor.submit_conversion(file, request)
        
        return ProcessingResponse(
            task_id=task_id,
            status=ProcessingStatus.QUEUED,
            message="Document conversion queued successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error converting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}")
async def get_conversion_status(task_id: str):
    """Get conversion task status"""
    try:
        status = await app.state.processor.get_task_status(task_id)
        return status
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=404, detail="Task not found")


@app.get("/result/{task_id}")
async def get_conversion_result(task_id: str):
    """Get conversion result"""
    try:
        result = await app.state.processor.get_task_result(task_id)
        return result
    except Exception as e:
        logger.error(f"Error getting task result: {str(e)}")
        raise HTTPException(status_code=404, detail="Result not found")


@app.get("/tasks")
async def list_conversion_tasks(
    limit: int = 10,
    offset: int = 0,
    status: Optional[ProcessingStatus] = None
):
    """List conversion tasks"""
    try:
        tasks = await app.state.processor.list_tasks(limit, offset, status)
        return tasks
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/task/{task_id}")
async def cancel_conversion_task(task_id: str):
    """Cancel a conversion task"""
    try:
        result = await app.state.processor.cancel_task(task_id)
        return {"message": "Task cancelled successfully", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}")
        raise HTTPException(status_code=404, detail="Task not found")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
