#!/usr/bin/env python3
"""
DEAAP Agent Orchestrator Service
Creates and manages LLM agents from validated document assets
"""

import asyncio
import json
import logging
import os
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import openai
import tiktoken
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://deaap:deaap@postgres:5432/deaap")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://vector-db:8757")
AUTHORIZATION_URL = os.getenv("AUTHORIZATION_URL", "http://authorization:8758")
MODEL_REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "/app/models")
AGENT_STORAGE_PATH = os.getenv("AGENT_STORAGE_PATH", "/app/agents")

# Agent parameters
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AgentStatus(Enum):
    CREATED = "created"
    BUILDING = "building"
    READY = "ready"
    DEPLOYED = "deployed"
    ERROR = "error"
    ARCHIVED = "archived"

class AgentType(Enum):
    RAG_ASSISTANT = "rag_assistant"
    INSTRUCTION_TUNED = "instruction_tuned"
    LORA_FINE_TUNED = "lora_fine_tuned"
    HYBRID = "hybrid"

class AgentDeploymentMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    EDGE = "edge"
    DISTRIBUTED = "distributed"

# Database Models
class AgentDefinition(Base):
    __tablename__ = "agent_definitions"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    agent_type = Column(String, nullable=False)
    status = Column(String, default=AgentStatus.CREATED.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Configuration
    model_config = Column(JSON)
    rag_config = Column(JSON)
    lora_config = Column(JSON)
    deployment_config = Column(JSON)
    
    # Data sources
    document_sources = Column(JSON)  # Document IDs and validation records
    vector_indices = Column(JSON)   # Vector DB index IDs
    lora_adapters = Column(JSON)    # LORA adapter paths
    
    # Validation & Traceability
    consensus_record = Column(String)  # Consensus manager validation ID
    authorization_token = Column(String)  # Authorization service token
    validation_history = Column(JSON)
    
    # Deployment
    deployment_endpoint = Column(String)
    deployment_status = Column(String)
    runtime_metrics = Column(JSON)

class AgentSession(Base):
    __tablename__ = "agent_sessions"
    
    id = Column(String, primary_key=True)
    agent_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Session state
    conversation_history = Column(JSON)
    context_state = Column(JSON)
    active = Column(Boolean, default=True)

# Pydantic Models
class AgentCreationRequest(BaseModel):
    name: str = Field(..., description="Human-readable agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: AgentType = Field(AgentType.RAG_ASSISTANT, description="Type of agent to create")
    
    # Data sources
    document_sources: List[str] = Field(..., description="Document IDs to include")
    consensus_validation_id: str = Field(..., description="Consensus manager validation ID")
    
    # Model configuration
    base_model: str = Field(DEFAULT_MODEL, description="Base LLM model")
    max_context_length: int = Field(MAX_CONTEXT_LENGTH, description="Maximum context length")
    temperature: float = Field(TEMPERATURE, description="Sampling temperature")
    
    # RAG configuration
    rag_config: Optional[Dict[str, Any]] = Field(None, description="RAG system configuration")
    
    # LORA configuration
    lora_config: Optional[Dict[str, Any]] = Field(None, description="LORA fine-tuning configuration")
    
    # Deployment configuration
    deployment_mode: AgentDeploymentMode = Field(AgentDeploymentMode.LOCAL, description="Deployment mode")
    deployment_config: Optional[Dict[str, Any]] = Field(None, description="Deployment-specific configuration")

class AgentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    agent_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    
    model_config: Dict[str, Any]
    rag_config: Optional[Dict[str, Any]]
    lora_config: Optional[Dict[str, Any]]
    deployment_config: Dict[str, Any]
    
    document_sources: List[str]
    consensus_record: Optional[str]
    deployment_endpoint: Optional[str]
    deployment_status: Optional[str]

class AgentInteractionRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID (optional for new sessions)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    stream: bool = Field(False, description="Enable streaming response")

class AgentInteractionResponse(BaseModel):
    response: str
    session_id: str
    context_used: Optional[Dict[str, Any]]
    sources: Optional[List[Dict[str, Any]]]
    tokens_used: Optional[int]

# FastAPI app
app = FastAPI(
    title="DEAAP Agent Orchestrator",
    description="LLM Agent Creation and Management Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis connection
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_client = await aioredis.from_url(REDIS_URL)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Ensure directories exist
    os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)
    os.makedirs(AGENT_STORAGE_PATH, exist_ok=True)
    
    logger.info("Agent Orchestrator service started")

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()

# Service integrations
class AuthorizationService:
    @staticmethod
    async def validate_document_access(document_ids: List[str], consensus_validation_id: str) -> bool:
        """Validate access to documents through authorization service"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "document_ids": document_ids,
                    "consensus_validation_id": consensus_validation_id
                }
                async with session.post(f"{AUTHORIZATION_URL}/validate-access", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("authorized", False)
                    return False
        except Exception as e:
            logger.error(f"Authorization validation failed: {e}")
            return False

class VectorDatabaseService:
    @staticmethod
    async def get_document_vectors(document_ids: List[str]) -> Dict[str, Any]:
        """Retrieve vector embeddings for documents"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"document_ids": document_ids}
                async with session.post(f"{VECTOR_DB_URL}/get-vectors", json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    return {}
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return {}
    
    @staticmethod
    async def create_agent_index(agent_id: str, document_vectors: Dict[str, Any]) -> str:
        """Create a dedicated vector index for the agent"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "agent_id": agent_id,
                    "vectors": document_vectors,
                    "index_config": {
                        "metric": "cosine",
                        "dimensions": 768  # Default embedding dimension
                    }
                }
                async with session.post(f"{VECTOR_DB_URL}/create-index", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("index_id", "")
                    return ""
        except Exception as e:
            logger.error(f"Vector index creation failed: {e}")
            return ""

class ModelRegistryService:
    @staticmethod
    async def get_lora_adapters(document_ids: List[str]) -> Dict[str, str]:
        """Retrieve LORA adapters for documents"""
        # In a real implementation, this would interface with the model registry
        # For now, return mock paths
        adapters = {}
        for doc_id in document_ids:
            adapter_path = f"{MODEL_REGISTRY_PATH}/lora/{doc_id}/adapter.safetensors"
            if os.path.exists(adapter_path):
                adapters[doc_id] = adapter_path
        return adapters

class AgentRuntimeService:
    @staticmethod
    async def deploy_agent(agent_def: AgentDefinition) -> str:
        """Deploy agent to runtime environment"""
        deployment_id = f"agent-{agent_def.id}-{int(time.time())}"
        
        # In a real implementation, this would:
        # 1. Package the agent configuration
        # 2. Deploy to the specified runtime (local/cloud/edge)
        # 3. Start the agent service
        # 4. Return the deployment endpoint
        
        # Mock deployment
        endpoint = f"http://agent-runtime:8800/agents/{deployment_id}"
        
        # Store deployment info
        agent_def.deployment_endpoint = endpoint
        agent_def.deployment_status = "deployed"
        
        return endpoint

# Agent building logic
async def build_rag_agent(agent_def: AgentDefinition, document_vectors: Dict[str, Any]) -> bool:
    """Build RAG-based agent"""
    try:
        # Create vector index for the agent
        vector_index_id = await VectorDatabaseService.create_agent_index(
            agent_def.id, document_vectors
        )
        
        if not vector_index_id:
            raise Exception("Failed to create vector index")
        
        # Update agent configuration
        agent_def.vector_indices = {"primary": vector_index_id}
        agent_def.rag_config = {
            "vector_index": vector_index_id,
            "similarity_threshold": 0.7,
            "max_context_docs": 5,
            "chunk_overlap": 100
        }
        
        return True
        
    except Exception as e:
        logger.error(f"RAG agent building failed: {e}")
        return False

async def build_lora_agent(agent_def: AgentDefinition, document_ids: List[str]) -> bool:
    """Build LORA fine-tuned agent"""
    try:
        # Get LORA adapters
        lora_adapters = await ModelRegistryService.get_lora_adapters(document_ids)
        
        if not lora_adapters:
            raise Exception("No LORA adapters found for documents")
        
        # Update agent configuration
        agent_def.lora_adapters = lora_adapters
        agent_def.lora_config = {
            "adapters": lora_adapters,
            "merge_strategy": "weighted_average",
            "scaling_factor": 1.0
        }
        
        return True
        
    except Exception as e:
        logger.error(f"LORA agent building failed: {e}")
        return False

async def build_hybrid_agent(agent_def: AgentDefinition, document_vectors: Dict[str, Any], document_ids: List[str]) -> bool:
    """Build hybrid RAG + LORA agent"""
    try:
        # Build both RAG and LORA components
        rag_success = await build_rag_agent(agent_def, document_vectors)
        lora_success = await build_lora_agent(agent_def, document_ids)
        
        if not (rag_success and lora_success):
            raise Exception("Failed to build hybrid components")
        
        # Merge configurations
        agent_def.model_config["hybrid_mode"] = True
        
        return True
        
    except Exception as e:
        logger.error(f"Hybrid agent building failed: {e}")
        return False

async def build_agent_background(agent_id: str):
    """Background task to build agent"""
    db = SessionLocal()
    try:
        agent_def = db.query(AgentDefinition).filter(AgentDefinition.id == agent_id).first()
        if not agent_def:
            return
        
        # Update status
        agent_def.status = AgentStatus.BUILDING.value
        db.commit()
        
        # Validate document access
        access_valid = await AuthorizationService.validate_document_access(
            agent_def.document_sources, agent_def.consensus_record
        )
        
        if not access_valid:
            raise Exception("Document access validation failed")
        
        # Get document vectors
        document_vectors = await VectorDatabaseService.get_document_vectors(
            agent_def.document_sources
        )
        
        # Build agent based on type
        build_success = False
        
        if agent_def.agent_type == AgentType.RAG_ASSISTANT.value:
            build_success = await build_rag_agent(agent_def, document_vectors)
        elif agent_def.agent_type == AgentType.LORA_FINE_TUNED.value:
            build_success = await build_lora_agent(agent_def, agent_def.document_sources)
        elif agent_def.agent_type == AgentType.HYBRID.value:
            build_success = await build_hybrid_agent(agent_def, document_vectors, agent_def.document_sources)
        
        if build_success:
            # Deploy agent
            endpoint = await AgentRuntimeService.deploy_agent(agent_def)
            agent_def.deployment_endpoint = endpoint
            agent_def.status = AgentStatus.READY.value
        else:
            agent_def.status = AgentStatus.ERROR.value
        
        db.commit()
        
        # Notify via Redis
        await redis_client.publish(
            f"agent:{agent_id}:status",
            json.dumps({"status": agent_def.status, "endpoint": agent_def.deployment_endpoint})
        )
        
    except Exception as e:
        logger.error(f"Agent building failed: {e}")
        if 'agent_def' in locals():
            agent_def.status = AgentStatus.ERROR.value
            db.commit()
    finally:
        db.close()

# API endpoints
@app.post("/agents", response_model=AgentResponse)
async def create_agent(
    request: AgentCreationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new agent from validated document assets"""
    
    # Generate agent ID
    agent_id = f"agent-{uuid.uuid4().hex[:8]}"
    
    # Create agent definition
    agent_def = AgentDefinition(
        id=agent_id,
        name=request.name,
        description=request.description,
        agent_type=request.agent_type.value,
        status=AgentStatus.CREATED.value,
        model_config={
            "base_model": request.base_model,
            "max_context_length": request.max_context_length,
            "temperature": request.temperature
        },
        rag_config=request.rag_config or {},
        lora_config=request.lora_config or {},
        deployment_config=request.deployment_config or {"mode": request.deployment_mode.value},
        document_sources=request.document_sources,
        consensus_record=request.consensus_validation_id
    )
    
    db.add(agent_def)
    db.commit()
    db.refresh(agent_def)
    
    # Start building in background
    background_tasks.add_task(build_agent_background, agent_id)
    
    return AgentResponse(
        id=agent_def.id,
        name=agent_def.name,
        description=agent_def.description,
        agent_type=agent_def.agent_type,
        status=agent_def.status,
        created_at=agent_def.created_at,
        updated_at=agent_def.updated_at,
        model_config=agent_def.model_config,
        rag_config=agent_def.rag_config,
        lora_config=agent_def.lora_config,
        deployment_config=agent_def.deployment_config,
        document_sources=agent_def.document_sources,
        consensus_record=agent_def.consensus_record,
        deployment_endpoint=agent_def.deployment_endpoint,
        deployment_status=agent_def.deployment_status
    )

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """Get agent details"""
    agent_def = db.query(AgentDefinition).filter(AgentDefinition.id == agent_id).first()
    if not agent_def:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return AgentResponse(
        id=agent_def.id,
        name=agent_def.name,
        description=agent_def.description,
        agent_type=agent_def.agent_type,
        status=agent_def.status,
        created_at=agent_def.created_at,
        updated_at=agent_def.updated_at,
        model_config=agent_def.model_config,
        rag_config=agent_def.rag_config,
        lora_config=agent_def.lora_config,
        deployment_config=agent_def.deployment_config,
        document_sources=agent_def.document_sources,
        consensus_record=agent_def.consensus_record,
        deployment_endpoint=agent_def.deployment_endpoint,
        deployment_status=agent_def.deployment_status
    )

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents(
    status: Optional[str] = None,
    agent_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List agents with optional filtering"""
    query = db.query(AgentDefinition)
    
    if status:
        query = query.filter(AgentDefinition.status == status)
    if agent_type:
        query = query.filter(AgentDefinition.agent_type == agent_type)
    
    agents = query.offset(offset).limit(limit).all()
    
    return [
        AgentResponse(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            agent_type=agent.agent_type,
            status=agent.status,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            model_config=agent.model_config,
            rag_config=agent.rag_config,
            lora_config=agent.lora_config,
            deployment_config=agent.deployment_config,
            document_sources=agent.document_sources,
            consensus_record=agent.consensus_record,
            deployment_endpoint=agent.deployment_endpoint,
            deployment_status=agent.deployment_status
        )
        for agent in agents
    ]

@app.post("/agents/{agent_id}/interact", response_model=AgentInteractionResponse)
async def interact_with_agent(
    agent_id: str,
    request: AgentInteractionRequest,
    db: Session = Depends(get_db)
):
    """Interact with a deployed agent"""
    agent_def = db.query(AgentDefinition).filter(AgentDefinition.id == agent_id).first()
    if not agent_def:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if agent_def.status != AgentStatus.READY.value:
        raise HTTPException(status_code=400, detail="Agent not ready for interaction")
    
    # Get or create session
    session_id = request.session_id or f"session-{uuid.uuid4().hex[:8]}"
    session = db.query(AgentSession).filter(AgentSession.id == session_id).first()
    
    if not session:
        session = AgentSession(
            id=session_id,
            agent_id=agent_id,
            user_id="user",  # TODO: Get from auth context
            conversation_history=[],
            context_state={}
        )
        db.add(session)
    
    # Update conversation history
    conversation_history = session.conversation_history or []
    conversation_history.append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Generate response (mock implementation)
    # In a real implementation, this would call the deployed agent
    response_text = f"Agent {agent_def.name} received: {request.message}"
    
    conversation_history.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    session.conversation_history = conversation_history
    session.last_activity = datetime.utcnow()
    
    db.commit()
    
    return AgentInteractionResponse(
        response=response_text,
        session_id=session_id,
        context_used=request.context,
        sources=[],
        tokens_used=len(request.message) + len(response_text)  # Mock token count
    )

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    """Delete an agent"""
    agent_def = db.query(AgentDefinition).filter(AgentDefinition.id == agent_id).first()
    if not agent_def:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Archive instead of deleting
    agent_def.status = AgentStatus.ARCHIVED.value
    agent_def.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Agent archived successfully"}

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str, db: Session = Depends(get_db)):
    """Get real-time agent status"""
    agent_def = db.query(AgentDefinition).filter(AgentDefinition.id == agent_id).first()
    if not agent_def:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent_id,
        "status": agent_def.status,
        "deployment_endpoint": agent_def.deployment_endpoint,
        "deployment_status": agent_def.deployment_status,
        "runtime_metrics": agent_def.runtime_metrics,
        "last_updated": agent_def.updated_at
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-orchestrator",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8759)
