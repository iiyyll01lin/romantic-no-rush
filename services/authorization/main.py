#!/usr/bin/env python3
"""
DEAAP Authorization Service
Manages access control and permissions for document usage and agent deployment
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://deaap_user:deaap_password@localhost:5432/authorization")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Configuration
BLOCKCHAIN_RPC = os.getenv("BLOCKCHAIN_RPC", "http://localhost:8545")
CONSENSUS_MANAGER_URL = os.getenv("CONSENSUS_MANAGER_URL", "http://localhost:8760")
JWT_SECRET = os.getenv("JWT_SECRET", "deaap-jwt-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# ============================================================================
# Database Models
# ============================================================================

class Permission(Base):
    __tablename__ = "permissions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    resource_type = Column(String, nullable=False)  # document, agent, model
    resource_id = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)  # read, write, deploy, delete
    granted_by = Column(String, nullable=False)
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime, nullable=True)
    revoked_by = Column(String, nullable=True)
    consensus_id = Column(String, nullable=True)  # Link to consensus decision
    metadata = Column(JSON, nullable=True)

class AccessRequest(Base):
    __tablename__ = "access_requests"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    action = Column(String, nullable=False)
    justification = Column(Text, nullable=True)
    business_unit = Column(String, nullable=False)
    priority = Column(String, default="normal")  # low, normal, high, urgent
    status = Column(String, default="pending")  # pending, approved, denied, expired
    requested_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    processed_by = Column(String, nullable=True)
    consensus_id = Column(String, nullable=True)
    approval_details = Column(JSON, nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    success = Column(Boolean, nullable=False)
    details = Column(JSON, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# ============================================================================
# Pydantic Models
# ============================================================================

class AuthUser(BaseModel):
    user_id: str
    business_unit: str
    roles: List[str]
    permissions: List[str]

class AccessRequestModel(BaseModel):
    resource_type: str
    resource_id: str
    action: str
    justification: Optional[str] = None
    priority: str = "normal"

class PermissionModel(BaseModel):
    resource_type: str
    resource_id: str
    action: str
    granted_by: str
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class AuthorizationCheck(BaseModel):
    resource_type: str
    resource_id: str
    action: str

class AuthorizationResult(BaseModel):
    authorized: bool
    reason: str
    permission_id: Optional[str] = None
    expires_at: Optional[datetime] = None

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="DEAAP Authorization Service",
    description="Access control and permissions management for decentralized AI agents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Redis client
redis_client = None

# ============================================================================
# Authentication & Authorization
# ============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> AuthUser:
    """Extract and validate JWT token to get current user"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        user_id = payload.get("user_id")
        business_unit = payload.get("business_unit")
        roles = payload.get("roles", [])
        permissions = payload.get("permissions", [])
        
        if not user_id or not business_unit:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        return AuthUser(
            user_id=user_id,
            business_unit=business_unit,
            roles=roles,
            permissions=permissions
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def log_access(db: Session, user_id: str, action: str, resource_type: str, 
                    resource_id: str, success: bool, details: Optional[Dict] = None):
    """Log access attempt for audit trail"""
    audit_log = AuditLog(
        id=f"audit-{int(time.time() * 1000000)}",
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        success=success,
        details=details
    )
    db.add(audit_log)
    db.commit()

# ============================================================================
# Core Authorization Logic
# ============================================================================

async def check_permission(db: Session, user_id: str, resource_type: str, 
                          resource_id: str, action: str) -> AuthorizationResult:
    """Check if user has permission for specific action on resource"""
    
    # Query for active permissions
    permission = db.query(Permission).filter(
        Permission.user_id == user_id,
        Permission.resource_type == resource_type,
        Permission.resource_id == resource_id,
        Permission.action == action,
        Permission.revoked == False,
        (Permission.expires_at.is_(None) | (Permission.expires_at > datetime.utcnow()))
    ).first()
    
    if permission:
        return AuthorizationResult(
            authorized=True,
            reason="Direct permission granted",
            permission_id=permission.id,
            expires_at=permission.expires_at
        )
    
    # Check for wildcard permissions (resource_id = "*")
    wildcard_permission = db.query(Permission).filter(
        Permission.user_id == user_id,
        Permission.resource_type == resource_type,
        Permission.resource_id == "*",
        Permission.action == action,
        Permission.revoked == False,
        (Permission.expires_at.is_(None) | (Permission.expires_at > datetime.utcnow()))
    ).first()
    
    if wildcard_permission:
        return AuthorizationResult(
            authorized=True,
            reason="Wildcard permission granted",
            permission_id=wildcard_permission.id,
            expires_at=wildcard_permission.expires_at
        )
    
    # Check for role-based permissions
    # TODO: Implement role-based access control
    
    return AuthorizationResult(
        authorized=False,
        reason="No permission found"
    )

async def request_consensus_authorization(user_id: str, request: AccessRequestModel) -> str:
    """Submit access request to consensus manager for multi-BU validation"""
    
    consensus_request = {
        "request_type": "access_authorization",
        "data": {
            "user_id": user_id,
            "resource_type": request.resource_type,
            "resource_id": request.resource_id,
            "action": request.action,
            "justification": request.justification,
            "priority": request.priority,
            "timestamp": datetime.utcnow().isoformat()
        },
        "validation_criteria": {
            "min_validators": 3,
            "consensus_threshold": 0.67,
            "timeout": 300
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{CONSENSUS_MANAGER_URL}/consensus/initiate",
            json=consensus_request
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["consensus_id"]
            else:
                raise HTTPException(status_code=500, detail="Failed to initiate consensus")

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection"""
    global redis_client
    redis_client = await aioredis.from_url(REDIS_URL)
    logger.info("Authorization service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Redis connection"""
    if redis_client:
        await redis_client.close()

@app.post("/auth/login")
async def login(user_id: str, business_unit: str, password: str):
    """Authenticate user and return JWT token"""
    # TODO: Implement proper authentication against identity provider
    # For now, creating a demo token
    
    payload = {
        "user_id": user_id,
        "business_unit": business_unit,
        "roles": ["user"],  # TODO: Get from identity provider
        "permissions": [],  # TODO: Get from identity provider
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600
    }

@app.post("/auth/check")
async def check_authorization(
    check: AuthorizationCheck,
    current_user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> AuthorizationResult:
    """Check if current user is authorized for specific action"""
    
    result = await check_permission(
        db, current_user.user_id, check.resource_type, 
        check.resource_id, check.action
    )
    
    # Log the access check
    await log_access(
        db, current_user.user_id, f"check_{check.action}",
        check.resource_type, check.resource_id, result.authorized,
        {"reason": result.reason}
    )
    
    return result

@app.post("/auth/request")
async def request_access(
    request: AccessRequestModel,
    current_user: AuthUser = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Request access to a resource through consensus validation"""
    
    # Create access request record
    access_request = AccessRequest(
        id=f"req-{int(time.time() * 1000000)}",
        user_id=current_user.user_id,
        resource_type=request.resource_type,
        resource_id=request.resource_id,
        action=request.action,
        justification=request.justification,
        business_unit=current_user.business_unit,
        priority=request.priority
    )
    
    db.add(access_request)
    db.commit()
    
    # Submit to consensus manager
    consensus_id = await request_consensus_authorization(current_user.user_id, request)
    
    # Update request with consensus ID
    access_request.consensus_id = consensus_id
    db.commit()
    
    # Log the request
    await log_access(
        db, current_user.user_id, "request_access",
        request.resource_type, request.resource_id, True,
        {"consensus_id": consensus_id}
    )
    
    return {
        "request_id": access_request.id,
        "consensus_id": consensus_id,
        "status": "submitted",
        "message": "Access request submitted for consensus validation"
    }

@app.post("/auth/grant")
async def grant_permission(
    permission: PermissionModel,
    target_user_id: str,
    current_user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Grant permission to a user (requires appropriate authority)"""
    
    # TODO: Check if current user has authority to grant permissions
    
    new_permission = Permission(
        id=f"perm-{int(time.time() * 1000000)}",
        user_id=target_user_id,
        resource_type=permission.resource_type,
        resource_id=permission.resource_id,
        action=permission.action,
        granted_by=current_user.user_id,
        expires_at=permission.expires_at,
        metadata=permission.metadata
    )
    
    db.add(new_permission)
    db.commit()
    
    # Log the permission grant
    await log_access(
        db, current_user.user_id, "grant_permission",
        permission.resource_type, permission.resource_id, True,
        {"target_user": target_user_id, "permission_id": new_permission.id}
    )
    
    return {
        "permission_id": new_permission.id,
        "message": "Permission granted successfully"
    }

@app.post("/auth/revoke/{permission_id}")
async def revoke_permission(
    permission_id: str,
    current_user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke a permission"""
    
    permission = db.query(Permission).filter(Permission.id == permission_id).first()
    if not permission:
        raise HTTPException(status_code=404, detail="Permission not found")
    
    # TODO: Check if current user has authority to revoke this permission
    
    permission.revoked = True
    permission.revoked_at = datetime.utcnow()
    permission.revoked_by = current_user.user_id
    db.commit()
    
    # Log the revocation
    await log_access(
        db, current_user.user_id, "revoke_permission",
        permission.resource_type, permission.resource_id, True,
        {"permission_id": permission_id}
    )
    
    return {"message": "Permission revoked successfully"}

@app.get("/auth/permissions")
async def list_permissions(
    current_user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List current user's permissions"""
    
    permissions = db.query(Permission).filter(
        Permission.user_id == current_user.user_id,
        Permission.revoked == False,
        (Permission.expires_at.is_(None) | (Permission.expires_at > datetime.utcnow()))
    ).all()
    
    return {
        "permissions": [
            {
                "id": p.id,
                "resource_type": p.resource_type,
                "resource_id": p.resource_id,
                "action": p.action,
                "granted_at": p.granted_at,
                "expires_at": p.expires_at,
                "metadata": p.metadata
            }
            for p in permissions
        ]
    }

@app.get("/auth/requests")
async def list_access_requests(
    current_user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List access requests for current user"""
    
    requests = db.query(AccessRequest).filter(
        AccessRequest.user_id == current_user.user_id
    ).order_by(AccessRequest.requested_at.desc()).limit(50).all()
    
    return {
        "requests": [
            {
                "id": r.id,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "action": r.action,
                "status": r.status,
                "requested_at": r.requested_at,
                "processed_at": r.processed_at,
                "consensus_id": r.consensus_id
            }
            for r in requests
        ]
    }

@app.post("/auth/webhook/consensus")
async def consensus_webhook(consensus_result: Dict[str, Any]):
    """Webhook to receive consensus decisions and update permissions"""
    
    consensus_id = consensus_result.get("consensus_id")
    if not consensus_id:
        raise HTTPException(status_code=400, detail="Missing consensus_id")
    
    db = SessionLocal()
    try:
        # Find the access request
        access_request = db.query(AccessRequest).filter(
            AccessRequest.consensus_id == consensus_id
        ).first()
        
        if not access_request:
            raise HTTPException(status_code=404, detail="Access request not found")
        
        # Update request status based on consensus
        if consensus_result.get("approved", False):
            access_request.status = "approved"
            access_request.processed_at = datetime.utcnow()
            access_request.approval_details = consensus_result
            
            # Create permission
            permission = Permission(
                id=f"perm-{int(time.time() * 1000000)}",
                user_id=access_request.user_id,
                resource_type=access_request.resource_type,
                resource_id=access_request.resource_id,
                action=access_request.action,
                granted_by="consensus",
                consensus_id=consensus_id,
                metadata={
                    "consensus_result": consensus_result,
                    "original_request": access_request.id
                }
            )
            db.add(permission)
            
        else:
            access_request.status = "denied"
            access_request.processed_at = datetime.utcnow()
            access_request.approval_details = consensus_result
        
        db.commit()
        
        # Notify user through Redis
        if redis_client:
            await redis_client.publish(
                f"user:{access_request.user_id}:notifications",
                json.dumps({
                    "type": "access_request_decision",
                    "request_id": access_request.id,
                    "status": access_request.status,
                    "consensus_id": consensus_id
                })
            )
        
        return {"message": "Consensus result processed"}
        
    finally:
        db.close()

@app.get("/auth/audit")
async def get_audit_logs(
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get audit logs (requires admin privileges)"""
    
    # TODO: Check if user has audit read permissions
    
    query = db.query(AuditLog).filter(AuditLog.user_id == current_user.user_id)
    
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    if resource_id:
        query = query.filter(AuditLog.resource_id == resource_id)
    if start_date:
        query = query.filter(AuditLog.timestamp >= start_date)
    if end_date:
        query = query.filter(AuditLog.timestamp <= end_date)
    
    logs = query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    
    return {
        "logs": [
            {
                "id": log.id,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "timestamp": log.timestamp,
                "success": log.success,
                "details": log.details
            }
            for log in logs
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "authorization",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8763, 
        reload=True,
        log_level="info"
    )
