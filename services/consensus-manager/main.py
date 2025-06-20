#!/usr/bin/env python3
"""
DEAAP Consensus Manager Service
Orchestrates multi-BU validation using Cartesi-inspired consensus patterns
"""

import asyncio
import json
import logging
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://deaap:deaap@postgres:5432/deaap")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
BLOCKCHAIN_RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", "http://blockchain:8545")
CARTESI_NODE_URL = os.getenv("CARTESI_NODE_URL", "http://cartesi-node:8080")
BU_VALIDATOR_BASE_URL = os.getenv("BU_VALIDATOR_BASE_URL", "http://bu-validator")

# Consensus parameters
MIN_VALIDATOR_COUNT = int(os.getenv("MIN_VALIDATOR_COUNT", "3"))
CONSENSUS_THRESHOLD = float(os.getenv("CONSENSUS_THRESHOLD", "0.67"))  # 67% agreement required
VALIDATION_TIMEOUT = int(os.getenv("VALIDATION_TIMEOUT", "300"))  # 5 minutes
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConsensusState(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ABSTAIN = "abstain"

@dataclass
class ValidationRequest:
    request_id: str
    data_hash: str
    bu_assignments: List[str]
    validation_criteria: Dict[str, Any]
    metadata: Dict[str, Any]
    timeout: int = VALIDATION_TIMEOUT

@dataclass
class BUValidationResponse:
    bu_id: str
    result: ValidationResult
    confidence: float
    reasoning: str
    signature: str
    timestamp: datetime

class ConsensusRequest(Base):
    __tablename__ = "consensus_requests"
    
    id = Column(String, primary_key=True)
    data_hash = Column(String, nullable=False, index=True)
    bu_assignments = Column(JSON)  # List of assigned BU IDs
    validation_criteria = Column(JSON)
    metadata = Column(JSON)
    state = Column(String, default=ConsensusState.PENDING.value)
    
    # Consensus tracking
    required_validators = Column(Integer)
    received_validations = Column(Integer, default=0)
    consensus_result = Column(String)  # APPROVED/REJECTED/INCONCLUSIVE
    confidence_score = Column(Float)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    timeout_at = Column(DateTime)
    
    # Results
    validation_responses = Column(JSON, default=list)
    final_attestation = Column(Text)  # Blockchain attestation
    cartesi_proof = Column(JSON)  # Cartesi computation proof

class BUValidation(Base):
    __tablename__ = "bu_validations"
    
    id = Column(String, primary_key=True)
    consensus_request_id = Column(String, nullable=False, index=True)
    bu_id = Column(String, nullable=False)
    result = Column(String)  # ValidationResult enum
    confidence = Column(Float)
    reasoning = Column(Text)
    signature = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime)
    cartesi_input_hash = Column(String)
    cartesi_output_hash = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(
    title="DEAAP Consensus Manager",
    description="Multi-BU validation orchestration with Cartesi-powered consensus",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ConsensusRequestModel(BaseModel):
    data_hash: str = Field(..., description="SHA256 hash of data to validate")
    bu_assignments: List[str] = Field(..., description="List of BU IDs assigned for validation")
    validation_criteria: Dict[str, Any] = Field(default_factory=dict, description="Validation parameters and rules")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context and metadata")
    timeout_minutes: int = Field(default=5, description="Validation timeout in minutes")

class ConsensusStatusModel(BaseModel):
    request_id: str
    state: str
    progress: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None

class BUValidationModel(BaseModel):
    bu_id: str
    result: str  # ValidationResult
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    signature: str

# Redis client
redis_client = None

async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(REDIS_URL)
    return redis_client

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ConsensusManager:
    """
    Core consensus orchestration logic adapted from Cartesi stock exchange patterns
    """
    
    def __init__(self):
        self.active_requests: Dict[str, ValidationRequest] = {}
        self.validation_cache: Dict[str, List[BUValidationResponse]] = {}
    
    async def initiate_consensus(self, request: ConsensusRequestModel) -> str:
        """
        Initiate a new consensus validation process
        """
        request_id = hashlib.sha256(
            f"{request.data_hash}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Store in database
        db = SessionLocal()
        try:
            consensus_req = ConsensusRequest(
                id=request_id,
                data_hash=request.data_hash,
                bu_assignments=request.bu_assignments,
                validation_criteria=request.validation_criteria,
                metadata=request.metadata,
                required_validators=len(request.bu_assignments),
                timeout_at=datetime.utcnow() + timedelta(minutes=request.timeout_minutes)
            )
            db.add(consensus_req)
            db.commit()
            
            # Create validation request object
            validation_req = ValidationRequest(
                request_id=request_id,
                data_hash=request.data_hash,
                bu_assignments=request.bu_assignments,
                validation_criteria=request.validation_criteria,
                metadata=request.metadata,
                timeout=request.timeout_minutes * 60
            )
            
            self.active_requests[request_id] = validation_req
            
            # Start validation process asynchronously
            asyncio.create_task(self._orchestrate_validation(validation_req))
            
            logger.info(f"Initiated consensus request {request_id} for data {request.data_hash}")
            return request_id
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to initiate consensus: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initiate consensus: {e}")
        finally:
            db.close()
    
    async def _orchestrate_validation(self, request: ValidationRequest):
        """
        Orchestrate the multi-BU validation process
        """
        logger.info(f"Starting validation orchestration for request {request.request_id}")
        
        try:
            # Update state to active
            await self._update_consensus_state(request.request_id, ConsensusState.ACTIVE)
            
            # Dispatch validation requests to BU validators
            validation_tasks = []
            for bu_id in request.bu_assignments:
                task = asyncio.create_task(
                    self._dispatch_bu_validation(request, bu_id)
                )
                validation_tasks.append(task)
            
            # Wait for validations with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*validation_tasks, return_exceptions=True),
                    timeout=request.timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Validation timeout for request {request.request_id}")
                await self._update_consensus_state(request.request_id, ConsensusState.TIMEOUT)
                return
            
            # Aggregate results and determine consensus
            await self._aggregate_consensus_results(request.request_id)
            
        except Exception as e:
            logger.error(f"Validation orchestration failed for {request.request_id}: {e}")
            await self._update_consensus_state(request.request_id, ConsensusState.FAILED)
    
    async def _dispatch_bu_validation(self, request: ValidationRequest, bu_id: str):
        """
        Send validation request to specific BU validator
        """
        logger.info(f"Dispatching validation to BU {bu_id} for request {request.request_id}")
        
        validation_payload = {
            "request_id": request.request_id,
            "data_hash": request.data_hash,
            "validation_criteria": request.validation_criteria,
            "metadata": request.metadata
        }
        
        try:
            # Determine BU validator URL (could be different ports/services per BU)
            bu_validator_url = f"{BU_VALIDATOR_BASE_URL}-{bu_id}:8755"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{bu_validator_url}/validate",
                    json=validation_payload,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        await self._record_bu_validation(request.request_id, bu_id, result)
                    else:
                        logger.error(f"BU {bu_id} validation failed with status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to dispatch validation to BU {bu_id}: {e}")
    
    async def _record_bu_validation(self, request_id: str, bu_id: str, validation_result: Dict[str, Any]):
        """
        Record validation result from a BU validator
        """
        db = SessionLocal()
        try:
            # Store individual BU validation
            bu_validation = BUValidation(
                id=f"{request_id}_{bu_id}",
                consensus_request_id=request_id,
                bu_id=bu_id,
                result=validation_result.get("result", "abstain"),
                confidence=validation_result.get("confidence", 0.0),
                reasoning=validation_result.get("reasoning", ""),
                signature=validation_result.get("signature", ""),
                submitted_at=datetime.utcnow()
            )
            db.add(bu_validation)
            
            # Update consensus request
            consensus_req = db.query(ConsensusRequest).filter(
                ConsensusRequest.id == request_id
            ).first()
            
            if consensus_req:
                consensus_req.received_validations += 1
                
                # Add to validation responses
                responses = consensus_req.validation_responses or []
                responses.append(validation_result)
                consensus_req.validation_responses = responses
            
            db.commit()
            logger.info(f"Recorded validation from BU {bu_id} for request {request_id}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record BU validation: {e}")
        finally:
            db.close()
    
    async def _aggregate_consensus_results(self, request_id: str):
        """
        Aggregate BU validation results and determine final consensus
        """
        db = SessionLocal()
        try:
            consensus_req = db.query(ConsensusRequest).filter(
                ConsensusRequest.id == request_id
            ).first()
            
            if not consensus_req:
                logger.error(f"Consensus request {request_id} not found")
                return
            
            responses = consensus_req.validation_responses or []
            
            if len(responses) < MIN_VALIDATOR_COUNT:
                logger.warning(f"Insufficient validators for consensus: {len(responses)} < {MIN_VALIDATOR_COUNT}")
                consensus_req.state = ConsensusState.FAILED.value
                consensus_req.consensus_result = "INSUFFICIENT_VALIDATORS"
                db.commit()
                return
            
            # Count votes
            approved_count = sum(1 for r in responses if r.get("result") == "approved")
            rejected_count = sum(1 for r in responses if r.get("result") == "rejected")
            total_votes = approved_count + rejected_count  # Exclude abstentions
            
            if total_votes == 0:
                consensus_result = "INCONCLUSIVE"
                confidence_score = 0.0
            else:
                approval_ratio = approved_count / total_votes
                
                if approval_ratio >= CONSENSUS_THRESHOLD:
                    consensus_result = "APPROVED"
                    confidence_score = approval_ratio
                elif (rejected_count / total_votes) >= CONSENSUS_THRESHOLD:
                    consensus_result = "REJECTED"
                    confidence_score = rejected_count / total_votes
                else:
                    consensus_result = "INCONCLUSIVE"
                    confidence_score = max(approval_ratio, rejected_count / total_votes)
            
            # Update consensus request
            consensus_req.state = ConsensusState.COMPLETED.value
            consensus_req.consensus_result = consensus_result
            consensus_req.confidence_score = confidence_score
            consensus_req.completed_at = datetime.utcnow()
            
            # Generate attestation (simplified - would use actual blockchain in production)
            attestation = await self._generate_consensus_attestation(consensus_req, responses)
            consensus_req.final_attestation = attestation
            
            db.commit()
            
            # Notify interested services about consensus result
            await self._notify_consensus_result(request_id, consensus_result, confidence_score)
            
            logger.info(f"Consensus completed for {request_id}: {consensus_result} (confidence: {confidence_score:.2f})")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to aggregate consensus results: {e}")
        finally:
            db.close()
    
    async def _generate_consensus_attestation(self, consensus_req: ConsensusRequest, responses: List[Dict]) -> str:
        """
        Generate cryptographic attestation of consensus result
        """
        # Create attestation data structure
        attestation_data = {
            "request_id": consensus_req.id,
            "data_hash": consensus_req.data_hash,
            "consensus_result": consensus_req.consensus_result,
            "confidence_score": consensus_req.confidence_score,
            "validator_count": len(responses),
            "timestamp": consensus_req.completed_at.isoformat(),
            "validation_summary": {
                "approved": sum(1 for r in responses if r.get("result") == "approved"),
                "rejected": sum(1 for r in responses if r.get("result") == "rejected"),
                "abstained": sum(1 for r in responses if r.get("result") == "abstain")
            }
        }
        
        # Generate hash of attestation (in production, this would be signed)
        attestation_json = json.dumps(attestation_data, sort_keys=True)
        attestation_hash = hashlib.sha256(attestation_json.encode()).hexdigest()
        
        return json.dumps({
            "attestation_data": attestation_data,
            "attestation_hash": attestation_hash,
            "signature": f"0x{attestation_hash}"  # Placeholder signature
        })
    
    async def _notify_consensus_result(self, request_id: str, result: str, confidence: float):
        """
        Notify other services about consensus completion
        """
        redis = await get_redis()
        
        notification = {
            "event": "consensus_completed",
            "request_id": request_id,
            "result": result,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await redis.publish("consensus_results", json.dumps(notification))
        logger.info(f"Published consensus result notification for {request_id}")
    
    async def _update_consensus_state(self, request_id: str, state: ConsensusState):
        """
        Update consensus request state
        """
        db = SessionLocal()
        try:
            consensus_req = db.query(ConsensusRequest).filter(
                ConsensusRequest.id == request_id
            ).first()
            
            if consensus_req:
                consensus_req.state = state.value
                if state == ConsensusState.ACTIVE:
                    consensus_req.started_at = datetime.utcnow()
                db.commit()
                
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update consensus state: {e}")
        finally:
            db.close()

# Global consensus manager instance
consensus_manager = ConsensusManager()

# API Endpoints
@app.post("/consensus/initiate", response_model=Dict[str, str])
async def initiate_consensus(request: ConsensusRequestModel):
    """
    Initiate a new consensus validation process
    """
    try:
        request_id = await consensus_manager.initiate_consensus(request)
        return {"request_id": request_id, "status": "initiated"}
    except Exception as e:
        logger.error(f"Failed to initiate consensus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/consensus/{request_id}/status", response_model=ConsensusStatusModel)
async def get_consensus_status(request_id: str, db: Session = Depends(get_db)):
    """
    Get the status of a consensus validation request
    """
    consensus_req = db.query(ConsensusRequest).filter(
        ConsensusRequest.id == request_id
    ).first()
    
    if not consensus_req:
        raise HTTPException(status_code=404, detail="Consensus request not found")
    
    # Calculate progress
    progress = {
        "received_validations": consensus_req.received_validations,
        "required_validators": consensus_req.required_validators,
        "completion_percentage": (consensus_req.received_validations / consensus_req.required_validators * 100) if consensus_req.required_validators > 0 else 0
    }
    
    results = None
    if consensus_req.state == ConsensusState.COMPLETED.value:
        results = {
            "consensus_result": consensus_req.consensus_result,
            "confidence_score": consensus_req.confidence_score,
            "final_attestation": consensus_req.final_attestation,
            "validation_responses": consensus_req.validation_responses
        }
    
    return ConsensusStatusModel(
        request_id=request_id,
        state=consensus_req.state,
        progress=progress,
        results=results
    )

@app.get("/consensus/{request_id}/results")
async def get_consensus_results(request_id: str, db: Session = Depends(get_db)):
    """
    Get detailed consensus validation results
    """
    consensus_req = db.query(ConsensusRequest).filter(
        ConsensusRequest.id == request_id
    ).first()
    
    if not consensus_req:
        raise HTTPException(status_code=404, detail="Consensus request not found")
    
    if consensus_req.state != ConsensusState.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Consensus not yet completed")
    
    # Get individual BU validations
    bu_validations = db.query(BUValidation).filter(
        BUValidation.consensus_request_id == request_id
    ).all()
    
    return {
        "request_id": request_id,
        "data_hash": consensus_req.data_hash,
        "consensus_result": consensus_req.consensus_result,
        "confidence_score": consensus_req.confidence_score,
        "validation_summary": {
            "total_validators": len(bu_validations),
            "individual_validations": [
                {
                    "bu_id": v.bu_id,
                    "result": v.result,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                    "submitted_at": v.submitted_at.isoformat()
                }
                for v in bu_validations
            ]
        },
        "attestation": json.loads(consensus_req.final_attestation) if consensus_req.final_attestation else None,
        "metadata": consensus_req.metadata,
        "timing": {
            "created_at": consensus_req.created_at.isoformat(),
            "started_at": consensus_req.started_at.isoformat() if consensus_req.started_at else None,
            "completed_at": consensus_req.completed_at.isoformat() if consensus_req.completed_at else None
        }
    }

@app.post("/consensus/{request_id}/bu-validation")
async def submit_bu_validation(
    request_id: str, 
    validation: BUValidationModel,
    db: Session = Depends(get_db)
):
    """
    Submit validation result from a BU validator (internal endpoint)
    """
    try:
        await consensus_manager._record_bu_validation(request_id, validation.bu_id, validation.dict())
        return {"status": "validation_recorded", "bu_id": validation.bu_id}
    except Exception as e:
        logger.error(f"Failed to submit BU validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/consensus/requests")
async def list_consensus_requests(
    limit: int = 50,
    offset: int = 0,
    state: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List consensus validation requests with optional filtering
    """
    query = db.query(ConsensusRequest)
    
    if state:
        query = query.filter(ConsensusRequest.state == state)
    
    requests = query.offset(offset).limit(limit).all()
    
    return {
        "requests": [
            {
                "id": req.id,
                "data_hash": req.data_hash,
                "state": req.state,
                "consensus_result": req.consensus_result,
                "confidence_score": req.confidence_score,
                "created_at": req.created_at.isoformat(),
                "progress": f"{req.received_validations}/{req.required_validators}"
            }
            for req in requests
        ],
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": query.count()
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Test database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        # Test Redis connection
        redis = await get_redis()
        await redis.ping()
        
        return {
            "status": "healthy",
            "service": "consensus-manager",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8760)
