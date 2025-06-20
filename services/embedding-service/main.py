#!/usr/bin/env python3
"""
DEAAP Embedding Service
Standalone embedding generation service for documents and queries
"""

import asyncio
import json
import logging
import os
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import aioredis
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://deaap:deaap@postgres:5432/deaap")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class EmbeddingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class EmbeddingRequest(Base):
    __tablename__ = "embedding_requests"
    
    id = Column(String, primary_key=True)
    request_type = Column(String, nullable=False)  # "document", "query", "batch"
    status = Column(String, default=EmbeddingStatus.PENDING.value)
    input_data = Column(JSON, nullable=False)
    embedding_results = Column(JSON)
    model_used = Column(String, nullable=False)
    vector_dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="DEAAP Embedding Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
embedding_model = None
redis_client = None

# Pydantic models
class EmbeddingRequestModel(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    model: Optional[str] = Field(default=None, description="Embedding model to use")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    include_metadata: bool = Field(default=False, description="Include metadata in response")

class BatchEmbeddingRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="List of documents with text and metadata")
    model: Optional[str] = Field(default=None, description="Embedding model to use")
    chunk_size: int = Field(default=512, description="Maximum chunk size for text splitting")
    overlap: int = Field(default=50, description="Overlap between chunks")

class EmbeddingResponse(BaseModel):
    request_id: str
    embeddings: List[List[float]]
    model_used: str
    vector_dimension: int
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None

class BatchEmbeddingResponse(BaseModel):
    request_id: str
    status: str
    total_documents: int
    completed_documents: int
    created_at: datetime
    estimated_completion: Optional[datetime] = None

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Model management
class EmbeddingModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        
    async def load_model(self, model_name: str = None):
        """Load or switch embedding model"""
        if model_name is None:
            model_name = EMBEDDING_MODEL
            
        if model_name in self.models:
            self.current_model = self.models[model_name]
            return self.current_model
            
        logger.info(f"Loading embedding model: {model_name}")
        try:
            # Load SentenceTransformer model
            model = SentenceTransformer(model_name, cache_folder=MODEL_CACHE_DIR)
            model.to(DEVICE)
            
            self.models[model_name] = model
            self.current_model = model
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    def get_model(self, model_name: str = None):
        """Get currently loaded model"""
        if model_name and model_name in self.models:
            return self.models[model_name]
        return self.current_model
    
    async def generate_embeddings(self, texts: List[str], model_name: str = None, normalize: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        model = self.get_model(model_name)
        if not model:
            model = await self.load_model(model_name)
        
        try:
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=min(BATCH_SIZE, len(batch_texts)),
                    show_progress_bar=False,
                    normalize_embeddings=normalize
                )
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

# Global model manager
model_manager = EmbeddingModelManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    global redis_client
    logger.info("Starting DEAAP Embedding Service...")
    
    # Initialize Redis
    redis_client = aioredis.from_url(REDIS_URL)
    
    # Load default embedding model
    await model_manager.load_model()
    
    logger.info("Embedding Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()
    logger.info("Embedding Service shut down")

# Utility functions
def split_text_into_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        start = end - overlap
    
    return chunks

async def process_batch_embedding_background(request_id: str, documents: List[Dict[str, Any]], 
                                           model_name: str, chunk_size: int, overlap: int):
    """Background task to process batch embeddings"""
    db = SessionLocal()
    try:
        request_record = db.query(EmbeddingRequest).filter(EmbeddingRequest.id == request_id).first()
        if not request_record:
            return
        
        # Update status
        request_record.status = EmbeddingStatus.PROCESSING.value
        db.commit()
        
        # Process documents
        results = []
        for i, doc in enumerate(documents):
            try:
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                # Split into chunks if needed
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                
                # Generate embeddings for chunks
                embeddings = await model_manager.generate_embeddings(chunks, model_name)
                
                # Store results
                doc_result = {
                    "document_id": doc.get("id", f"doc_{i}"),
                    "chunks": len(chunks),
                    "embeddings": embeddings.tolist(),
                    "metadata": metadata
                }
                results.append(doc_result)
                
                # Update progress in Redis
                await redis_client.set(
                    f"embedding_progress:{request_id}",
                    json.dumps({"completed": i + 1, "total": len(documents)}),
                    ex=3600
                )
                
            except Exception as e:
                logger.error(f"Failed to process document {i}: {e}")
                continue
        
        # Save results
        request_record.embedding_results = {"documents": results}
        request_record.status = EmbeddingStatus.COMPLETED.value
        request_record.completed_at = datetime.utcnow()
        db.commit()
        
        # Notify completion
        await redis_client.publish(
            f"embedding_complete:{request_id}",
            json.dumps({"status": "completed", "results_count": len(results)})
        )
        
    except Exception as e:
        logger.error(f"Batch embedding processing failed: {e}")
        request_record.status = EmbeddingStatus.ERROR.value
        request_record.error_message = str(e)
        request_record.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()

# API endpoints
@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequestModel, db: Session = Depends(get_db)):
    """Generate embeddings for a list of texts"""
    request_id = f"embed-{int(time.time())}-{hashlib.md5(str(request.texts).encode()).hexdigest()[:8]}"
    
    start_time = time.time()
    
    try:
        # Generate embeddings
        embeddings = await model_manager.generate_embeddings(
            request.texts, 
            request.model, 
            request.normalize
        )
        
        processing_time = time.time() - start_time
        
        # Store request record
        model_used = request.model or EMBEDDING_MODEL
        embedding_request = EmbeddingRequest(
            id=request_id,
            request_type="query",
            status=EmbeddingStatus.COMPLETED.value,
            input_data={"texts": request.texts, "model": model_used},
            embedding_results={"embeddings": embeddings.tolist()},
            model_used=model_used,
            vector_dimension=embeddings.shape[1],
            completed_at=datetime.utcnow()
        )
        
        db.add(embedding_request)
        db.commit()
        
        # Prepare response metadata
        metadata = None
        if request.include_metadata:
            metadata = {
                "texts_count": len(request.texts),
                "model_info": {
                    "name": model_used,
                    "device": DEVICE,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH
                },
                "processing_stats": {
                    "batch_size": BATCH_SIZE,
                    "total_tokens": sum(len(text.split()) for text in request.texts)
                }
            }
        
        return EmbeddingResponse(
            request_id=request_id,
            embeddings=embeddings.tolist(),
            model_used=model_used,
            vector_dimension=embeddings.shape[1],
            processing_time=processing_time,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def batch_embed_documents(
    request: BatchEmbeddingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Process batch document embedding (async)"""
    request_id = f"batch-{int(time.time())}-{hashlib.md5(str(request.documents).encode()).hexdigest()[:8]}"
    
    # Create request record
    embedding_request = EmbeddingRequest(
        id=request_id,
        request_type="batch",
        status=EmbeddingStatus.PENDING.value,
        input_data={
            "documents_count": len(request.documents),
            "model": request.model or EMBEDDING_MODEL,
            "chunk_size": request.chunk_size,
            "overlap": request.overlap
        },
        model_used=request.model or EMBEDDING_MODEL,
        vector_dimension=VECTOR_DIMENSION
    )
    
    db.add(embedding_request)
    db.commit()
    
    # Start background processing
    background_tasks.add_task(
        process_batch_embedding_background,
        request_id,
        request.documents,
        request.model or EMBEDDING_MODEL,
        request.chunk_size,
        request.overlap
    )
    
    return BatchEmbeddingResponse(
        request_id=request_id,
        status=EmbeddingStatus.PENDING.value,
        total_documents=len(request.documents),
        completed_documents=0,
        created_at=datetime.utcnow()
    )

@app.get("/embed/batch/{request_id}")
async def get_batch_status(request_id: str, db: Session = Depends(get_db)):
    """Get batch embedding status"""
    request_record = db.query(EmbeddingRequest).filter(EmbeddingRequest.id == request_id).first()
    if not request_record:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Get progress from Redis
    progress_data = await redis_client.get(f"embedding_progress:{request_id}")
    progress = json.loads(progress_data) if progress_data else {"completed": 0, "total": 0}
    
    return {
        "request_id": request_id,
        "status": request_record.status,
        "total_documents": request_record.input_data.get("documents_count", 0),
        "completed_documents": progress.get("completed", 0),
        "created_at": request_record.created_at,
        "completed_at": request_record.completed_at,
        "error_message": request_record.error_message
    }

@app.get("/embed/batch/{request_id}/results")
async def get_batch_results(request_id: str, db: Session = Depends(get_db)):
    """Get batch embedding results"""
    request_record = db.query(EmbeddingRequest).filter(EmbeddingRequest.id == request_id).first()
    if not request_record:
        raise HTTPException(status_code=404, detail="Request not found")
    
    if request_record.status != EmbeddingStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Request not completed yet")
    
    return {
        "request_id": request_id,
        "status": request_record.status,
        "results": request_record.embedding_results,
        "model_used": request_record.model_used,
        "vector_dimension": request_record.vector_dimension,
        "completed_at": request_record.completed_at
    }

@app.get("/models")
async def list_available_models():
    """List available embedding models"""
    return {
        "current_model": EMBEDDING_MODEL,
        "loaded_models": list(model_manager.models.keys()),
        "supported_models": [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2", 
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "text-embedding-ada-002"  # OpenAI model (if API key provided)
        ],
        "device": DEVICE,
        "vector_dimension": VECTOR_DIMENSION
    }

@app.post("/models/load")
async def load_model(model_name: str):
    """Load a specific embedding model"""
    try:
        await model_manager.load_model(model_name)
        return {
            "message": f"Model {model_name} loaded successfully",
            "model_name": model_name,
            "device": DEVICE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.get("/similarity")
async def compute_similarity(text1: str, text2: str, model: Optional[str] = None):
    """Compute similarity between two texts"""
    try:
        embeddings = await model_manager.generate_embeddings([text1, text2], model)
        
        # Compute cosine similarity
        similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                          (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        
        return {
            "text1": text1,
            "text2": text2,
            "similarity": similarity,
            "model_used": model or EMBEDDING_MODEL
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model_manager.current_model else "not_loaded"
    
    return {
        "status": "healthy",
        "service": "embedding-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "model_status": model_status,
        "current_model": EMBEDDING_MODEL,
        "device": DEVICE,
        "vector_dimension": VECTOR_DIMENSION
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8756)
