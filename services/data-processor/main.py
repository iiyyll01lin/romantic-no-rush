#!/usr/bin/env python3
"""
DEAAP Data Processor Service
Bridges document processing outputs with consensus validation system
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://deaap:deaap@postgres:5432/deaap")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DOC_INGESTER_URL = os.getenv("DOC_INGESTER_URL", "http://doc-ingester:8752")
CONSENSUS_MANAGER_URL = os.getenv("CONSENSUS_MANAGER_URL", "http://consensus-manager:8760")
AGENT_ORCHESTRATOR_URL = os.getenv("AGENT_ORCHESTRATOR_URL", "http://agent-orchestrator:8770")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False)
    bu_list = Column(JSON)  # List of BUs for validation
    status = Column(String, default="pending")
    metadata = Column(JSON)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    error_message = Column(Text)

Base.metadata.create_all(bind=engine)

# Pydantic models
class ProcessingRequest(BaseModel):
    document_id: str
    bu_list: List[str] = Field(description="List of BU names for validation")
    processing_config: Dict[str, Any] = Field(default_factory=dict)
    consensus_config: Dict[str, Any] = Field(default_factory=dict)

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: str
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class DataAsset(BaseModel):
    asset_id: str
    document_id: str
    chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    synthetic_data: List[Dict[str, Any]]
    lora_adaptors: Dict[str, Any]
    metadata: Dict[str, Any]

# FastAPI app
app = FastAPI(title="DEAAP Data Processor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_client = await aioredis.from_url(REDIS_URL)
    logger.info("Data Processor Service started")

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DataProcessor:
    def __init__(self):
        self.session = aiohttp.ClientSession()
    
    async def process_document_to_assets(self, job_id: str, request: ProcessingRequest) -> DataAsset:
        """Process document through complete pipeline to create data assets"""
        logger.info(f"Processing document {request.document_id} for job {job_id}")
        
        # Step 1: Get processed document data from doc-ingester
        doc_data = await self._get_document_data(request.document_id)
        
        # Step 2: Validate data structure and quality
        validated_data = await self._validate_data_quality(doc_data)
        
        # Step 3: Create structured data assets
        data_asset = await self._create_data_assets(validated_data, request.document_id)
        
        # Step 4: Generate metadata for consensus validation
        consensus_metadata = await self._generate_consensus_metadata(data_asset, request.bu_list)
        data_asset.metadata.update(consensus_metadata)
        
        logger.info(f"Successfully created data assets for document {request.document_id}")
        return data_asset
    
    async def _get_document_data(self, document_id: str) -> Dict[str, Any]:
        """Retrieve processed document data from doc-ingester"""
        async with self.session.get(f"{DOC_INGESTER_URL}/documents/{document_id}/processed") as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"Failed to get document data: {response.status}")
            return await response.json()
    
    async def _validate_data_quality(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and completeness"""
        required_fields = ["chunks", "embeddings", "synthetic_data", "metadata"]
        missing_fields = [field for field in required_fields if field not in doc_data]
        
        if missing_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required fields: {missing_fields}"
            )
        
        # Quality checks
        chunks = doc_data.get("chunks", [])
        embeddings = doc_data.get("embeddings", [])
        
        if len(chunks) != len(embeddings):
            raise HTTPException(
                status_code=400,
                detail="Mismatch between chunks and embeddings count"
            )
        
        # Validate chunk quality
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk.get("content", "").strip()) > 50:  # Minimum content length
                valid_chunks.append(chunk)
        
        if len(valid_chunks) < len(chunks) * 0.8:  # At least 80% valid chunks
            raise HTTPException(
                status_code=400,
                detail="Document quality too low - insufficient valid chunks"
            )
        
        doc_data["chunks"] = valid_chunks
        doc_data["embeddings"] = embeddings[:len(valid_chunks)]
        
        return doc_data
    
    async def _create_data_assets(self, validated_data: Dict[str, Any], document_id: str) -> DataAsset:
        """Create structured data assets from validated data"""
        asset_id = f"asset_{document_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Process LORA adaptors if available
        lora_adaptors = validated_data.get("lora_adaptors", {})
        if not lora_adaptors and validated_data.get("synthetic_data"):
            # Generate basic LORA metadata if not present
            lora_adaptors = {
                "type": "basic",
                "training_data_size": len(validated_data.get("synthetic_data", [])),
                "created_at": datetime.utcnow().isoformat(),
                "status": "ready_for_training"
            }
        
        return DataAsset(
            asset_id=asset_id,
            document_id=document_id,
            chunks=validated_data["chunks"],
            embeddings=validated_data["embeddings"],
            synthetic_data=validated_data.get("synthetic_data", []),
            lora_adaptors=lora_adaptors,
            metadata={
                "created_at": datetime.utcnow().isoformat(),
                "quality_score": self._calculate_quality_score(validated_data),
                "processing_version": "1.0.0",
                "original_metadata": validated_data.get("metadata", {})
            }
        )
    
    async def _generate_consensus_metadata(self, data_asset: DataAsset, bu_list: List[str]) -> Dict[str, Any]:
        """Generate metadata required for consensus validation"""
        return {
            "consensus_required": True,
            "target_bus": bu_list,
            "data_hash": self._calculate_data_hash(data_asset),
            "validation_rules": {
                "min_approvals": max(2, len(bu_list) // 2 + 1),
                "timeout_seconds": 3600,
                "required_bu_categories": ["legal", "security"] if "sensitive" in str(data_asset.metadata) else []
            }
        }
    
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        score = 0.0
        
        # Chunk quality (40%)
        chunks = data.get("chunks", [])
        if chunks:
            avg_chunk_length = sum(len(chunk.get("content", "")) for chunk in chunks) / len(chunks)
            score += min(0.4, avg_chunk_length / 1000)  # Normalize to 0.4 max
        
        # Embedding quality (30%)
        embeddings = data.get("embeddings", [])
        if embeddings and len(embeddings) == len(chunks):
            score += 0.3
        
        # Synthetic data quality (20%)
        synthetic_data = data.get("synthetic_data", [])
        if synthetic_data and len(synthetic_data) >= len(chunks) * 0.5:
            score += 0.2
        
        # Metadata completeness (10%)
        metadata = data.get("metadata", {})
        if len(metadata) >= 3:  # At least 3 metadata fields
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_data_hash(self, data_asset: DataAsset) -> str:
        """Calculate hash for consensus validation"""
        import hashlib
        
        # Create deterministic hash from key data
        hash_content = {
            "chunks_count": len(data_asset.chunks),
            "embeddings_count": len(data_asset.embeddings),
            "synthetic_count": len(data_asset.synthetic_data),
            "asset_id": data_asset.asset_id
        }
        
        hash_string = json.dumps(hash_content, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

# Global processor instance
processor = DataProcessor()

@app.post("/process", response_model=Dict[str, str])
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start document processing pipeline"""
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.document_id}"
    
    # Create job record
    db = next(get_db())
    job = ProcessingJob(
        id=job_id,
        document_id=request.document_id,
        bu_list=request.bu_list,
        status="processing",
        metadata={"processing_config": request.processing_config}
    )
    db.add(job)
    db.commit()
    
    # Start background processing
    background_tasks.add_task(process_document_pipeline, job_id, request)
    
    return {"job_id": job_id, "status": "started"}

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get processing job status"""
    db = next(get_db())
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get progress from Redis if available
    progress_key = f"progress:{job_id}"
    progress_data = await redis_client.hgetall(progress_key)
    
    progress = float(progress_data.get("progress", 0) if progress_data else 0)
    current_step = progress_data.get("step", "unknown") if progress_data else "unknown"
    
    return ProcessingStatus(
        job_id=job_id,
        status=job.status,
        progress=progress,
        current_step=current_step,
        results=job.results,
        error_message=job.error_message
    )

@app.get("/assets/{job_id}", response_model=DataAsset)
async def get_data_assets(job_id: str):
    """Get processed data assets"""
    db = next(get_db())
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job.results:
        raise HTTPException(status_code=404, detail="No assets found")
    
    return DataAsset(**job.results)

async def process_document_pipeline(job_id: str, request: ProcessingRequest):
    """Background task for complete document processing pipeline"""
    db = next(get_db())
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    try:
        # Update progress
        await update_progress(job_id, 10, "starting_processing")
        
        # Step 1: Process document to data assets
        await update_progress(job_id, 30, "creating_data_assets")
        data_asset = await processor.process_document_to_assets(job_id, request)
        
        # Step 2: Initiate consensus validation
        await update_progress(job_id, 60, "initiating_consensus")
        consensus_result = await initiate_consensus_validation(data_asset, request.bu_list, request.consensus_config)
        
        # Step 3: Create AI agent if consensus approved
        await update_progress(job_id, 80, "creating_agent")
        agent_result = None
        if consensus_result.get("approved", False):
            agent_result = await create_ai_agent(data_asset, consensus_result)
        
        # Step 4: Complete and save results
        await update_progress(job_id, 100, "completed")
        
        results = {
            **data_asset.dict(),
            "consensus_result": consensus_result,
            "agent_result": agent_result
        }
        
        job.status = "completed"
        job.results = results
        job.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Processing completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {str(e)}")
        job.status = "failed"
        job.error_message = str(e)
        job.updated_at = datetime.utcnow()
        db.commit()
        
        await update_progress(job_id, 0, "failed")

async def update_progress(job_id: str, progress: float, step: str):
    """Update job progress in Redis"""
    progress_key = f"progress:{job_id}"
    await redis_client.hset(progress_key, mapping={
        "progress": str(progress),
        "step": step,
        "timestamp": datetime.utcnow().isoformat()
    })
    await redis_client.expire(progress_key, 3600)  # Expire after 1 hour

async def initiate_consensus_validation(data_asset: DataAsset, bu_list: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Initiate consensus validation with BU validators"""
    payload = {
        "data_hash": data_asset.metadata["data_hash"],
        "bu_list": bu_list,
        "asset_metadata": {
            "asset_id": data_asset.asset_id,
            "document_id": data_asset.document_id,
            "quality_score": data_asset.metadata.get("quality_score"),
            "chunks_count": len(data_asset.chunks),
            "synthetic_count": len(data_asset.synthetic_data)
        },
        "validation_config": config
    }
    
    async with processor.session.post(f"{CONSENSUS_MANAGER_URL}/validate", json=payload) as response:
        if response.status != 200:
            raise HTTPException(status_code=500, detail="Consensus validation failed")
        return await response.json()

async def create_ai_agent(data_asset: DataAsset, consensus_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create AI agent with validated data assets"""
    payload = {
        "asset_id": data_asset.asset_id,
        "chunks": data_asset.chunks,
        "embeddings": data_asset.embeddings,
        "synthetic_data": data_asset.synthetic_data,
        "lora_adaptors": data_asset.lora_adaptors,
        "consensus_proof": consensus_result.get("proof"),
        "authorized_users": consensus_result.get("authorized_users", [])
    }
    
    async with processor.session.post(f"{AGENT_ORCHESTRATOR_URL}/agents", json=payload) as response:
        if response.status != 201:
            raise HTTPException(status_code=500, detail="Agent creation failed")
        return await response.json()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8758)
