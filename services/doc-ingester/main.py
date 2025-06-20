import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import ProcessingRequest, ProcessingResponse, TaskStatus
from api.task_manager import TaskManager
from api.database import init_db
from api.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Document Ingester Service...")
    await init_db()
    
    # Initialize task manager
    app.state.task_manager = TaskManager()
    await app.state.task_manager.start()
    
    logger.info("Document Ingester Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Ingester Service...")
    await app.state.task_manager.stop()
    logger.info("Document Ingester Service stopped")


# Create FastAPI application
app = FastAPI(
    title="DEAAP Document Ingester",
    description="Document processing service for the Decentralized Enterprise AI Agent Platform",
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
    return {"status": "healthy", "service": "doc-ingester"}


@app.post("/upload", response_model=ProcessingResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_method: str = Form(default="semantic"),
    chunk_size: int = Form(default=2100),
    chunk_overlap: int = Form(default=200),
    generate_synthetic: bool = Form(default=True),
    generate_embeddings: bool = Form(default=True)
):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        max_size = settings.MAX_FILE_SIZE_BYTES
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            )
        
        # Reset file pointer
        await file.seek(0)
        
        # Create processing request
        request = ProcessingRequest(
            filename=file.filename,
            file_content=content,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            generate_synthetic=generate_synthetic,
            generate_embeddings=generate_embeddings
        )
        
        # Submit task
        task_id = await app.state.task_manager.submit_task(request)
        
        return ProcessingResponse(
            task_id=task_id,
            status="submitted",
            message="Document processing started"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get processing task status"""
    try:
        status = await app.state.task_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks(
    limit: int = 10,
    offset: int = 0,
    status: str = None
):
    """List processing tasks"""
    try:
        tasks = await app.state.task_manager.list_tasks(
            limit=limit,
            offset=offset,
            status_filter=status
        )
        return {"tasks": tasks}
        
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a processing task"""
    try:
        success = await app.state.task_manager.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
        
        return {"message": "Task cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    try:
        metrics = await app.state.task_manager.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8752,
        reload=os.getenv("DEVELOPMENT_MODE", "false").lower() == "true",
        log_level=settings.LOG_LEVEL.lower()
    )
