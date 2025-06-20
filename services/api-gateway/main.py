#!/usr/bin/env python3
"""
DEAAP API Gateway
Central routing and authentication hub for all microservices
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
JWT_SECRET = os.getenv("JWT_SECRET", "deaap-jwt-secret-key")
JWT_ALGORITHM = "HS256"

# Service URLs
SERVICES = {
    "authorization": os.getenv("AUTHORIZATION_SERVICE_URL", "http://authorization:8763"),
    "doc-ingester": os.getenv("DOC_INGESTER_URL", "http://doc-ingester:8752"),
    "yy-chunker": os.getenv("YY_CHUNKER_URL", "http://yy-chunker:8754"),
    "consensus-manager": os.getenv("CONSENSUS_MANAGER_URL", "http://consensus-manager:8760"),
    "agent-orchestrator": os.getenv("AGENT_ORCHESTRATOR_URL", "http://agent-orchestrator:8770"),
    "vector-database": os.getenv("VECTOR_DATABASE_URL", "http://vector-database:8774"),
    "data-processor": os.getenv("DATA_PROCESSOR_URL", "http://data-processor:8751"),
    "bu-validator": os.getenv("BU_VALIDATOR_URL", "http://bu-validator:8762"),
    "llm-runtime": os.getenv("LLM_RUNTIME_URL", "http://llm-runtime:8772"),
    "rag-engine": os.getenv("RAG_ENGINE_URL", "http://rag-engine:8773"),
}

# Route mappings
ROUTE_MAPPINGS = {
    "/api/auth": "authorization",
    "/api/documents": "doc-ingester", 
    "/api/chunks": "yy-chunker",
    "/api/consensus": "consensus-manager",
    "/api/agents": "agent-orchestrator",
    "/api/vectors": "vector-database",
    "/api/data": "data-processor",
    "/api/validators": "bu-validator",
    "/api/llm": "llm-runtime",
    "/api/rag": "rag-engine",
}

# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/api/auth/login",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json"
}

# ============================================================================
# Pydantic Models
# ============================================================================

class ServiceStatus(BaseModel):
    service: str
    status: str
    response_time_ms: Optional[float] = None
    last_check: datetime

class SystemStatus(BaseModel):
    overall_status: str
    services: List[ServiceStatus]
    timestamp: datetime

class RouteMetrics(BaseModel):
    route: str
    method: str
    count: int
    avg_response_time: float
    error_rate: float

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="DEAAP API Gateway",
    description="Central routing and authentication hub for decentralized AI agents",
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
service_health_cache = {}

# ============================================================================
# Authentication & Authorization
# ============================================================================

def extract_user_from_token(token: str) -> Dict[str, Any]:
    """Extract user information from JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {
            "user_id": payload.get("user_id"),
            "business_unit": payload.get("business_unit"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", [])
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def verify_authentication(request: Request) -> Optional[Dict[str, Any]]:
    """Verify authentication from request headers"""
    
    # Skip authentication for public endpoints
    path = request.url.path
    if path in PUBLIC_ENDPOINTS or path.startswith("/docs") or path.startswith("/redoc"):
        return None
    
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = auth_header.split(" ")[1]
    user_info = extract_user_from_token(token)
    
    return user_info

# ============================================================================
# Service Discovery & Health Checks
# ============================================================================

async def check_service_health(service_name: str, service_url: str) -> ServiceStatus:
    """Check health of a specific service"""
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{service_url}/health") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    status = "healthy"
                else:
                    status = "unhealthy"
                    
                return ServiceStatus(
                    service=service_name,
                    status=status,
                    response_time_ms=response_time,
                    last_check=datetime.utcnow()
                )
                
    except Exception as e:
        logger.warning(f"Health check failed for {service_name}: {e}")
        return ServiceStatus(
            service=service_name,
            status="unhealthy",
            response_time_ms=None,
            last_check=datetime.utcnow()
        )

async def check_all_services_health() -> SystemStatus:
    """Check health of all services"""
    
    health_checks = []
    for service_name, service_url in SERVICES.items():
        health_checks.append(check_service_health(service_name, service_url))
    
    service_statuses = await asyncio.gather(*health_checks, return_exceptions=True)
    
    # Filter out exceptions and count healthy services
    valid_statuses = [s for s in service_statuses if isinstance(s, ServiceStatus)]
    healthy_count = sum(1 for s in valid_statuses if s.status == "healthy")
    
    # Determine overall status
    if healthy_count == len(valid_statuses):
        overall_status = "healthy"
    elif healthy_count > len(valid_statuses) * 0.5:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return SystemStatus(
        overall_status=overall_status,
        services=valid_statuses,
        timestamp=datetime.utcnow()
    )

# ============================================================================
# Request Routing & Proxying
# ============================================================================

def find_target_service(path: str) -> Optional[str]:
    """Find the target service for a given path"""
    for route_prefix, service_name in ROUTE_MAPPINGS.items():
        if path.startswith(route_prefix):
            return service_name
    return None

async def proxy_request(
    request: Request,
    target_service: str,
    user_info: Optional[Dict[str, Any]] = None
) -> Response:
    """Proxy request to target service"""
    
    service_url = SERVICES.get(target_service)
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Service {target_service} not found")
    
    # Build target URL
    path = request.url.path
    query_string = str(request.url.query)
    target_url = f"{service_url}{path}"
    if query_string:
        target_url += f"?{query_string}"
    
    # Prepare headers
    headers = dict(request.headers)
    
    # Add user context if authenticated
    if user_info:
        headers["X-User-ID"] = user_info["user_id"]
        headers["X-Business-Unit"] = user_info["business_unit"]
        headers["X-User-Roles"] = ",".join(user_info["roles"])
    
    # Remove hop-by-hop headers
    hop_by_hop_headers = ["connection", "keep-alive", "proxy-authenticate", 
                         "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"]
    for header in hop_by_hop_headers:
        headers.pop(header, None)
    
    start_time = time.time()
    
    try:
        # Read request body
        body = await request.body()
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                response_time = (time.time() - start_time) * 1000
                
                # Log request for metrics
                await log_request_metrics(
                    path=path,
                    method=request.method,
                    service=target_service,
                    status_code=response.status,
                    response_time=response_time,
                    user_id=user_info.get("user_id") if user_info else None
                )
                
                # Build response
                content = await response.read()
                response_headers = dict(response.headers)
                
                # Remove hop-by-hop headers from response
                for header in hop_by_hop_headers:
                    response_headers.pop(header, None)
                
                return Response(
                    content=content,
                    status_code=response.status,
                    headers=response_headers,
                    media_type=response_headers.get("content-type")
                )
                
    except aiohttp.ClientTimeout:
        await log_request_metrics(path, request.method, target_service, 504, 
                                 (time.time() - start_time) * 1000, 
                                 user_info.get("user_id") if user_info else None)
        raise HTTPException(status_code=504, detail="Service timeout")
        
    except aiohttp.ClientError as e:
        await log_request_metrics(path, request.method, target_service, 502, 
                                 (time.time() - start_time) * 1000,
                                 user_info.get("user_id") if user_info else None)
        raise HTTPException(status_code=502, detail=f"Service error: {e}")

async def log_request_metrics(path: str, method: str, service: str, 
                            status_code: int, response_time: float, user_id: Optional[str]):
    """Log request metrics to Redis for analytics"""
    if not redis_client:
        return
    
    try:
        metric = {
            "path": path,
            "method": method,
            "service": service,
            "status_code": status_code,
            "response_time": response_time,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in Redis list (for recent metrics)
        await redis_client.lpush("request_metrics", json.dumps(metric))
        await redis_client.ltrim("request_metrics", 0, 9999)  # Keep last 10k metrics
        
        # Update counters
        date_key = datetime.utcnow().strftime("%Y-%m-%d")
        await redis_client.hincrby(f"daily_stats:{date_key}", f"{service}:{method}:{status_code}", 1)
        await redis_client.expire(f"daily_stats:{date_key}", 86400 * 7)  # Keep for 7 days
        
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection and start background tasks"""
    global redis_client
    
    try:
        redis_client = await aioredis.from_url(REDIS_URL)
        logger.info("API Gateway started successfully")
        
        # Start background health check task
        asyncio.create_task(periodic_health_check())
        
    except Exception as e:
        logger.error(f"Failed to initialize API Gateway: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Redis connection"""
    if redis_client:
        await redis_client.close()

async def periodic_health_check():
    """Periodic health check of all services"""
    while True:
        try:
            system_status = await check_all_services_health()
            if redis_client:
                await redis_client.setex(
                    "system_health", 60, 
                    json.dumps(system_status.dict(), default=str)
                )
            await asyncio.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            await asyncio.sleep(30)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_all_requests(request: Request, path: str):
    """Main proxy endpoint that routes all requests"""
    
    # Verify authentication
    user_info = await verify_authentication(request)
    
    # Find target service
    full_path = f"/{path}"
    target_service = find_target_service(full_path)
    
    if not target_service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Proxy the request
    return await proxy_request(request, target_service, user_info)

@app.get("/health")
async def gateway_health():
    """API Gateway health check"""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/system/health")
async def system_health():
    """Get overall system health status"""
    return await check_all_services_health()

@app.get("/api/system/metrics")
async def system_metrics():
    """Get system metrics and analytics"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Metrics service unavailable")
    
    try:
        # Get recent metrics
        recent_metrics_raw = await redis_client.lrange("request_metrics", 0, 99)
        recent_metrics = [json.loads(m) for m in recent_metrics_raw]
        
        # Get daily stats
        date_key = datetime.utcnow().strftime("%Y-%m-%d")
        daily_stats = await redis_client.hgetall(f"daily_stats:{date_key}")
        daily_stats = {k.decode(): int(v) for k, v in daily_stats.items()}
        
        # Calculate route metrics
        route_metrics = {}
        for metric in recent_metrics:
            route_key = f"{metric['path']}:{metric['method']}"
            if route_key not in route_metrics:
                route_metrics[route_key] = {
                    "count": 0,
                    "total_time": 0,
                    "error_count": 0
                }
            
            route_metrics[route_key]["count"] += 1
            route_metrics[route_key]["total_time"] += metric["response_time"]
            if metric["status_code"] >= 400:
                route_metrics[route_key]["error_count"] += 1
        
        # Convert to RouteMetrics objects
        formatted_route_metrics = []
        for route_key, stats in route_metrics.items():
            path, method = route_key.split(":", 1)
            avg_response_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            error_rate = stats["error_count"] / stats["count"] if stats["count"] > 0 else 0
            
            formatted_route_metrics.append(RouteMetrics(
                route=path,
                method=method,
                count=stats["count"],
                avg_response_time=avg_response_time,
                error_rate=error_rate
            ))
        
        return {
            "recent_requests": len(recent_metrics),
            "daily_stats": daily_stats,
            "route_metrics": formatted_route_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/api/system/services")
async def list_services():
    """List all registered services"""
    return {
        "services": [
            {
                "name": name,
                "url": url,
                "routes": [route for route, service in ROUTE_MAPPINGS.items() if service == name]
            }
            for name, url in SERVICES.items()
        ]
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8765, 
        reload=True,
        log_level="info"
    )
