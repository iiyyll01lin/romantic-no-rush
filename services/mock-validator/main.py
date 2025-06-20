#!/usr/bin/env python3
"""
DEAAP Mock Validator Service
Development mode validator for rapid testing
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONSENSUS_MANAGER_URL = os.getenv("CONSENSUS_MANAGER_URL", "http://consensus-manager:8760")
VALIDATION_MODE = os.getenv("VALIDATION_MODE", "mock")
APPROVAL_RATE = float(os.getenv("APPROVAL_RATE", "0.8"))  # 80% approval rate
RESPONSE_DELAY = int(os.getenv("RESPONSE_DELAY", "5"))  # 5 second delay
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ============================================================================
# Pydantic Models
# ============================================================================

class ValidationRequest(BaseModel):
    consensus_id: str
    request_type: str
    data: Dict[str, Any]
    validation_criteria: Dict[str, Any]

class ValidationResponse(BaseModel):
    consensus_id: str
    validator_id: str
    approved: bool
    confidence: float
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None

class ValidatorStatus(BaseModel):
    validator_id: str
    status: str
    uptime: float
    processed_requests: int
    approval_rate: float

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="DEAAP Mock Validator Service",
    description="Development mode validator for rapid testing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
redis_client = None
start_time = time.time()
processed_requests = 0
approved_requests = 0

# ============================================================================
# Mock Validation Logic
# ============================================================================

def generate_mock_validation(request: ValidationRequest) -> ValidationResponse:
    """Generate a mock validation response"""
    global processed_requests, approved_requests
    
    processed_requests += 1
    
    # Simulate random approval based on configured rate
    approved = random.random() < APPROVAL_RATE
    if approved:
        approved_requests += 1
    
    # Generate confidence score
    confidence = random.uniform(0.7, 0.95) if approved else random.uniform(0.3, 0.6)
    
    # Generate reasoning based on request type
    if request.request_type == "access_authorization":
        if approved:
            reasoning = f"Mock validator approves access request. User business unit alignment verified (mock)."
        else:
            reasoning = f"Mock validator denies access request. Insufficient justification (mock simulation)."
    elif request.request_type == "data_usage":
        if approved:
            reasoning = f"Mock validator approves data usage. Privacy compliance verified (mock)."
        else:
            reasoning = f"Mock validator denies data usage. Privacy concerns identified (mock simulation)."
    elif request.request_type == "agent_deployment":
        if approved:
            reasoning = f"Mock validator approves agent deployment. Model quality verified (mock)."
        else:
            reasoning = f"Mock validator denies agent deployment. Quality threshold not met (mock simulation)."
    else:
        if approved:
            reasoning = f"Mock validator approves general request. All criteria met (mock)."
        else:
            reasoning = f"Mock validator denies general request. Criteria not satisfied (mock simulation)."
    
    return ValidationResponse(
        consensus_id=request.consensus_id,
        validator_id="mock-validator-dev",
        approved=approved,
        confidence=confidence,
        reasoning=reasoning,
        metadata={
            "validation_mode": "mock",
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": RESPONSE_DELAY,
            "mock_parameters": {
                "approval_rate": APPROVAL_RATE,
                "response_delay": RESPONSE_DELAY
            }
        }
    )

async def submit_validation_to_consensus(response: ValidationResponse):
    """Submit validation response to consensus manager"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONSENSUS_MANAGER_URL}/consensus/{response.consensus_id}/validation",
                json=response.dict()
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Submitted validation for consensus {response.consensus_id}")
                else:
                    logger.error(f"Failed to submit validation: {resp.status}")
    except Exception as e:
        logger.error(f"Error submitting validation: {e}")

# ============================================================================
# Background Tasks
# ============================================================================

async def process_validation_request(request: ValidationRequest):
    """Process a validation request with mock delay"""
    logger.info(f"Processing validation request {request.consensus_id}")
    
    # Simulate processing delay
    await asyncio.sleep(RESPONSE_DELAY)
    
    # Generate mock validation
    response = generate_mock_validation(request)
    
    # Submit to consensus manager
    await submit_validation_to_consensus(response)
    
    logger.info(f"Completed validation for {request.consensus_id}: {'APPROVED' if response.approved else 'DENIED'}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection and register with consensus manager"""
    global redis_client
    
    try:
        redis_client = await aioredis.from_url(REDIS_URL)
        logger.info("Mock validator started")
        
        # Register with consensus manager
        await register_with_consensus_manager()
        
    except Exception as e:
        logger.error(f"Failed to initialize mock validator: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Redis connection"""
    if redis_client:
        await redis_client.close()

async def register_with_consensus_manager():
    """Register this validator with the consensus manager"""
    try:
        validator_info = {
            "validator_id": "mock-validator-dev",
            "validator_type": "mock",
            "business_unit": "development",
            "capabilities": ["access_authorization", "data_usage", "agent_deployment"],
            "endpoint": "http://mock-validator:8762",
            "metadata": {
                "mode": "development",
                "approval_rate": APPROVAL_RATE,
                "response_delay": RESPONSE_DELAY
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONSENSUS_MANAGER_URL}/validators/register",
                json=validator_info
            ) as response:
                if response.status == 200:
                    logger.info("Successfully registered with consensus manager")
                else:
                    logger.warning(f"Failed to register with consensus manager: {response.status}")
                    
    except Exception as e:
        logger.warning(f"Could not register with consensus manager: {e}")

@app.post("/validate")
async def validate_request(
    request: ValidationRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Receive and process a validation request"""
    
    logger.info(f"Received validation request for consensus {request.consensus_id}")
    
    # Add to background processing
    background_tasks.add_task(process_validation_request, request)
    
    return {
        "message": "Validation request received",
        "consensus_id": request.consensus_id,
        "validator_id": "mock-validator-dev",
        "estimated_completion": datetime.utcnow() + timedelta(seconds=RESPONSE_DELAY)
    }

@app.get("/status")
async def get_validator_status() -> ValidatorStatus:
    """Get current validator status"""
    
    uptime = time.time() - start_time
    current_approval_rate = approved_requests / processed_requests if processed_requests > 0 else 0
    
    return ValidatorStatus(
        validator_id="mock-validator-dev",
        status="active",
        uptime=uptime,
        processed_requests=processed_requests,
        approval_rate=current_approval_rate
    )

@app.post("/configure")
async def configure_validator(
    approval_rate: Optional[float] = None,
    response_delay: Optional[int] = None
):
    """Configure mock validator parameters"""
    global APPROVAL_RATE, RESPONSE_DELAY
    
    changes = {}
    
    if approval_rate is not None:
        if 0.0 <= approval_rate <= 1.0:
            APPROVAL_RATE = approval_rate
            changes["approval_rate"] = approval_rate
        else:
            raise HTTPException(status_code=400, detail="Approval rate must be between 0.0 and 1.0")
    
    if response_delay is not None:
        if response_delay >= 0:
            RESPONSE_DELAY = response_delay
            changes["response_delay"] = response_delay
        else:
            raise HTTPException(status_code=400, detail="Response delay must be non-negative")
    
    logger.info(f"Validator configuration updated: {changes}")
    
    return {
        "message": "Configuration updated",
        "changes": changes,
        "current_config": {
            "approval_rate": APPROVAL_RATE,
            "response_delay": RESPONSE_DELAY
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get validator metrics"""
    
    uptime = time.time() - start_time
    current_approval_rate = approved_requests / processed_requests if processed_requests > 0 else 0
    
    return {
        "validator_id": "mock-validator-dev",
        "uptime_seconds": uptime,
        "total_requests": processed_requests,
        "approved_requests": approved_requests,
        "denied_requests": processed_requests - approved_requests,
        "approval_rate": current_approval_rate,
        "configured_approval_rate": APPROVAL_RATE,
        "average_response_time": RESPONSE_DELAY,
        "last_activity": datetime.utcnow().isoformat()
    }

@app.post("/simulate/burst")
async def simulate_validation_burst(
    count: int = 10,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Simulate a burst of validation requests for testing"""
    
    if count > 100:
        raise HTTPException(status_code=400, detail="Maximum burst size is 100")
    
    requests_generated = []
    
    for i in range(count):
        # Generate mock validation request
        mock_request = ValidationRequest(
            consensus_id=f"mock-consensus-{int(time.time())}-{i}",
            request_type=random.choice(["access_authorization", "data_usage", "agent_deployment"]),
            data={
                "user_id": f"test-user-{i}",
                "resource_id": f"test-resource-{i}",
                "action": "read",
                "justification": f"Mock test request {i}"
            },
            validation_criteria={
                "min_validators": 1,
                "consensus_threshold": 0.5,
                "timeout": 60
            }
        )
        
        # Process in background
        background_tasks.add_task(process_validation_request, mock_request)
        requests_generated.append(mock_request.consensus_id)
    
    return {
        "message": f"Generated {count} mock validation requests",
        "consensus_ids": requests_generated,
        "estimated_completion": datetime.utcnow() + timedelta(seconds=RESPONSE_DELAY + 5)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mock-validator",
        "mode": VALIDATION_MODE,
        "uptime": time.time() - start_time,
        "processed_requests": processed_requests,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8762, 
        reload=True,
        log_level="info"
    )
