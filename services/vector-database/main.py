#!/usr/bin/env python3
"""
DEAAP Vector Database Service
Unified vector storage and retrieval for RAG systems
"""

import asyncio
import json
import logging
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://deaap_user:deaap_password@localhost:5432/deaap")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "vector-db")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
AUTHORIZATION_SERVICE_URL = os.getenv("AUTHORIZATION_SERVICE_URL", "http://authorization:8763")

# ============================================================================
# Database Models
# ============================================================================

class VectorCollection(Base):
    __tablename__ = "vector_collections"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    owner_id = Column(String, nullable=False)
    business_unit = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    vector_dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    document_count = Column(Integer, default=0)
    is_public = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)

class DocumentVector(Base):
    __tablename__ = "document_vectors"
    
    id = Column(String, primary_key=True)
    collection_id = Column(String, nullable=False, index=True)
    document_id = Column(String, nullable=False, index=True)
    chunk_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    source_file = Column(String, nullable=True)
    source_page = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=True)
    embedding_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# ============================================================================
# Pydantic Models
# ============================================================================

class VectorCollectionModel(BaseModel):
    name: str
    description: Optional[str] = None
    business_unit: str
    embedding_model: str = EMBEDDING_MODEL
    is_public: bool = False
    metadata: Optional[Dict[str, Any]] = None

class DocumentVectorModel(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    source_file: Optional[str] = None
    source_page: Optional[int] = None
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class VectorSearchQuery(BaseModel):
    query_text: str
    collection_name: str
    top_k: int = 10
    score_threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None

class VectorSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    source_file: Optional[str] = None
    source_page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class BulkInsertRequest(BaseModel):
    collection_name: str
    documents: List[DocumentVectorModel]

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="DEAAP Vector Database Service",
    description="Unified vector storage and retrieval for RAG systems",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
qdrant_client = None
embedding_model = None
redis_client = None

# ============================================================================
# Database Dependencies
# ============================================================================

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def check_authorization(user_id: str, action: str, resource_type: str, resource_id: str) -> bool:
    """Check user authorization through authorization service"""
    try:
        async with aiohttp.ClientSession() as session:
            auth_check = {
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action
            }
            headers = {"Authorization": f"Bearer {user_id}"}  # Simplified for demo
            
            async with session.post(
                f"{AUTHORIZATION_SERVICE_URL}/auth/check",
                json=auth_check,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("authorized", False)
                return False
    except:
        logger.error("Failed to check authorization")
        return False

# ============================================================================
# Vector Operations
# ============================================================================

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using the configured model"""
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")
    
    embeddings = embedding_model.encode([text])
    return embeddings[0].tolist()

def generate_embedding_hash(embedding: List[float]) -> str:
    """Generate a hash for the embedding for deduplication"""
    embedding_bytes = np.array(embedding).tobytes()
    return hashlib.sha256(embedding_bytes).hexdigest()

async def create_qdrant_collection(collection_name: str, vector_dimension: int):
    """Create a new collection in Qdrant"""
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create vector collection: {e}")

async def insert_vectors_to_qdrant(collection_name: str, vectors: List[Dict]):
    """Insert vectors into Qdrant collection"""
    try:
        points = []
        for vector_data in vectors:
            point = PointStruct(
                id=vector_data["chunk_id"],
                vector=vector_data["embedding"],
                payload={
                    "document_id": vector_data["document_id"],
                    "content": vector_data["content"],
                    "source_file": vector_data.get("source_file"),
                    "source_page": vector_data.get("source_page"),
                    "chunk_index": vector_data.get("chunk_index"),
                    "metadata": vector_data.get("metadata", {})
                }
            )
            points.append(point)
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Inserted {len(points)} vectors to collection {collection_name}")
        
    except Exception as e:
        logger.error(f"Failed to insert vectors to Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to insert vectors: {e}")

async def search_vectors_in_qdrant(collection_name: str, query_vector: List[float], 
                                  top_k: int, score_threshold: float, 
                                  filters: Optional[Dict] = None) -> List[VectorSearchResult]:
    """Search for similar vectors in Qdrant"""
    try:
        search_filter = None
        if filters:
            # Convert filters to Qdrant filter format
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                search_filter = Filter(must=conditions)
        
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter
        )
        
        results = []
        for result in search_results:
            search_result = VectorSearchResult(
                chunk_id=str(result.id),
                document_id=result.payload["document_id"],
                content=result.payload["content"],
                score=result.score,
                source_file=result.payload.get("source_file"),
                source_page=result.payload.get("source_page"),
                metadata=result.payload.get("metadata", {})
            )
            results.append(search_result)
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to search vectors in Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Qdrant client and embedding model"""
    global qdrant_client, embedding_model, redis_client
    
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        
        # Initialize embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        
        # Initialize Redis client
        redis_client = await aioredis.from_url(REDIS_URL)
        logger.info("Connected to Redis")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections"""
    if redis_client:
        await redis_client.close()

@app.post("/collections")
async def create_collection(
    collection: VectorCollectionModel,
    user_id: str,  # Simplified - should come from JWT
    db: Session = Depends(get_db)
):
    """Create a new vector collection"""
    
    # Check if collection name already exists
    existing = db.query(VectorCollection).filter(VectorCollection.name == collection.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Collection name already exists")
    
    # Create database record
    new_collection = VectorCollection(
        id=f"col-{int(time.time() * 1000000)}",
        name=collection.name,
        description=collection.description,
        owner_id=user_id,
        business_unit=collection.business_unit,
        embedding_model=collection.embedding_model,
        vector_dimension=VECTOR_DIMENSION,
        is_public=collection.is_public,
        metadata=collection.metadata
    )
    
    db.add(new_collection)
    db.commit()
    
    # Create Qdrant collection
    await create_qdrant_collection(collection.name, VECTOR_DIMENSION)
    
    return {
        "collection_id": new_collection.id,
        "name": new_collection.name,
        "message": "Collection created successfully"
    }

@app.get("/collections")
async def list_collections(
    user_id: str,  # Simplified - should come from JWT
    business_unit: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List accessible vector collections"""
    
    query = db.query(VectorCollection)
    
    # Filter by business unit if specified
    if business_unit:
        query = query.filter(VectorCollection.business_unit == business_unit)
    
    # Show public collections or owned collections
    query = query.filter(
        (VectorCollection.is_public == True) | (VectorCollection.owner_id == user_id)
    )
    
    collections = query.all()
    
    return {
        "collections": [
            {
                "id": col.id,
                "name": col.name,
                "description": col.description,
                "business_unit": col.business_unit,
                "document_count": col.document_count,
                "created_at": col.created_at,
                "is_public": col.is_public
            }
            for col in collections
        ]
    }

@app.post("/collections/{collection_name}/documents")
async def add_document_vectors(
    collection_name: str,
    documents: List[DocumentVectorModel],
    user_id: str,  # Simplified - should come from JWT
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Add document vectors to a collection"""
    
    # Check collection exists and user has access
    collection = db.query(VectorCollection).filter(VectorCollection.name == collection_name).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Check authorization
    authorized = await check_authorization(user_id, "write", "collection", collection_name)
    if not authorized and collection.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to add to this collection")
    
    # Process documents
    vector_data = []
    db_records = []
    
    for doc in documents:
        # Generate embedding
        embedding = generate_embedding(doc.content)
        embedding_hash = generate_embedding_hash(embedding)
        
        # Check for duplicates
        existing = db.query(DocumentVector).filter(
            DocumentVector.collection_id == collection.id,
            DocumentVector.embedding_hash == embedding_hash
        ).first()
        
        if existing:
            logger.info(f"Skipping duplicate content for chunk {doc.chunk_id}")
            continue
        
        # Prepare vector data for Qdrant
        vector_data.append({
            "chunk_id": doc.chunk_id,
            "document_id": doc.document_id,
            "content": doc.content,
            "embedding": embedding,
            "source_file": doc.source_file,
            "source_page": doc.source_page,
            "chunk_index": doc.chunk_index,
            "metadata": doc.metadata
        })
        
        # Prepare database record
        db_record = DocumentVector(
            id=f"vec-{int(time.time() * 1000000)}-{len(db_records)}",
            collection_id=collection.id,
            document_id=doc.document_id,
            chunk_id=doc.chunk_id,
            content=doc.content,
            source_file=doc.source_file,
            source_page=doc.source_page,
            chunk_index=doc.chunk_index,
            embedding_hash=embedding_hash,
            metadata=doc.metadata
        )
        db_records.append(db_record)
    
    if not vector_data:
        return {"message": "No new vectors to add (all duplicates)"}
    
    # Insert into database
    db.add_all(db_records)
    
    # Update collection count
    collection.document_count += len(db_records)
    collection.updated_at = datetime.utcnow()
    
    db.commit()
    
    # Insert into Qdrant in background
    background_tasks.add_task(insert_vectors_to_qdrant, collection_name, vector_data)
    
    return {
        "message": f"Added {len(vector_data)} vectors to collection {collection_name}",
        "processed": len(vector_data),
        "skipped": len(documents) - len(vector_data)
    }

@app.post("/collections/{collection_name}/search")
async def search_vectors(
    collection_name: str,
    search_query: VectorSearchQuery,
    user_id: str,  # Simplified - should come from JWT
    db: Session = Depends(get_db)
) -> List[VectorSearchResult]:
    """Search for similar vectors in a collection"""
    
    # Check collection exists and user has access
    collection = db.query(VectorCollection).filter(VectorCollection.name == collection_name).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Check authorization
    authorized = await check_authorization(user_id, "read", "collection", collection_name)
    if not authorized and not collection.is_public and collection.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to search this collection")
    
    # Generate query embedding
    query_embedding = generate_embedding(search_query.query_text)
    
    # Search in Qdrant
    results = await search_vectors_in_qdrant(
        collection_name=collection_name,
        query_vector=query_embedding,
        top_k=search_query.top_k,
        score_threshold=search_query.score_threshold,
        filters=search_query.filters
    )
    
    # Log search for analytics
    if redis_client:
        await redis_client.lpush(
            f"search_log:{collection_name}",
            json.dumps({
                "user_id": user_id,
                "query": search_query.query_text,
                "results_count": len(results),
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        await redis_client.ltrim(f"search_log:{collection_name}", 0, 999)  # Keep last 1000 searches
    
    return results

@app.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    user_id: str,  # Simplified - should come from JWT
    db: Session = Depends(get_db)
):
    """Delete a vector collection"""
    
    # Check collection exists and user owns it
    collection = db.query(VectorCollection).filter(VectorCollection.name == collection_name).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    if collection.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Only collection owner can delete")
    
    # Delete from Qdrant
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception as e:
        logger.warning(f"Failed to delete Qdrant collection {collection_name}: {e}")
    
    # Delete from database
    db.query(DocumentVector).filter(DocumentVector.collection_id == collection.id).delete()
    db.delete(collection)
    db.commit()
    
    return {"message": f"Collection {collection_name} deleted successfully"}

@app.get("/collections/{collection_name}/stats")
async def get_collection_stats(
    collection_name: str,
    user_id: str,  # Simplified - should come from JWT
    db: Session = Depends(get_db)
):
    """Get statistics for a vector collection"""
    
    # Check collection exists and user has access
    collection = db.query(VectorCollection).filter(VectorCollection.name == collection_name).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Check authorization
    authorized = await check_authorization(user_id, "read", "collection", collection_name)
    if not authorized and not collection.is_public and collection.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this collection")
    
    # Get document counts by document_id
    document_stats = db.query(
        DocumentVector.document_id,
        db.func.count(DocumentVector.id).label("chunk_count")
    ).filter(
        DocumentVector.collection_id == collection.id
    ).group_by(DocumentVector.document_id).all()
    
    # Get Qdrant collection info
    try:
        qdrant_info = qdrant_client.get_collection(collection_name)
        vector_count = qdrant_info.points_count
    except:
        vector_count = 0
    
    return {
        "collection_name": collection_name,
        "total_documents": len(document_stats),
        "total_chunks": collection.document_count,
        "vector_count": vector_count,
        "embedding_model": collection.embedding_model,
        "vector_dimension": collection.vector_dimension,
        "created_at": collection.created_at,
        "updated_at": collection.updated_at,
        "document_breakdown": [
            {"document_id": stat.document_id, "chunk_count": stat.chunk_count}
            for stat in document_stats
        ]
    }

@app.post("/bulk-insert")
async def bulk_insert_vectors(
    request: BulkInsertRequest,
    user_id: str,  # Simplified - should come from JWT
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Bulk insert vectors for high-throughput scenarios"""
    
    return await add_document_vectors(
        collection_name=request.collection_name,
        documents=request.documents,
        user_id=user_id,
        background_tasks=background_tasks,
        db=db
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        qdrant_info = qdrant_client.get_collections()
        qdrant_status = "healthy"
    except:
        qdrant_status = "unhealthy"
    
    return {
        "status": "healthy",
        "service": "vector-database",
        "qdrant_status": qdrant_status,
        "embedding_model": EMBEDDING_MODEL,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8774, 
        reload=True,
        log_level="info"
    )
