# System Design Specification 3: LLM Agent Orchestration & Authorization

## Overview

This specification defines the LLM agent orchestration system that assembles validated data assets (LORA adaptors and embeddings) into domain-specific AI agents based on authorization consensus.

## Component Architecture

```mermaid
graph TD
    subgraph "Agent Request Layer"
        UI[User Interface]
        API[Agent API Gateway]
        REQ[Request Processor]
        UI --> API
        API --> REQ
    end
    
    subgraph "Agent Orchestration"
        AO[Agent Orchestrator]
        AC[Agent Configurator]
        AM[Agent Manager]
        REQ --> AO
        AO --> AC
        AO --> AM
    end
    
    subgraph "Authorization Integration"
        AS[Authorization Service]
        PC[Permission Controller]
        AL[Audit Logger]
        AO --> AS
        AS --> PC
        AS --> AL
    end
    
    subgraph "Asset Management"
        LM[LORA Manager]
        EM[Embedding Manager]
        VDB[Vector Database]
        FM[File Manager]
        AC --> LM
        AC --> EM
        AC --> VDB
        LM --> FM
        EM --> FM
    end
    
    subgraph "Runtime Environment"
        LLM[LLM Runtime]
        RAG[RAG Engine]
        CACHE[Response Cache]
        MON[Monitor Service]
        AM --> LLM
        AM --> RAG
        LLM --> CACHE
        RAG --> CACHE
        AM --> MON
    end
    
    subgraph "Storage Layer"
        MDB[Model Database]
        VEC[Vector Store]
        META[Metadata Store]
        AUDIT[Audit Store]
        LM --> MDB
        EM --> VEC
        AC --> META
        AL --> AUDIT
    end
    
    class UI,API,REQ fill:#e1f5fe
    class AO,AC,AM fill:#f3e5f5
    class AS,PC,AL fill:#e8f5e8
    class LM,EM,VDB,FM fill:#fff3e0
    class LLM,RAG,CACHE,MON fill:#ffebee
    class MDB,VEC,META,AUDIT fill:#f0f0f0
```

## Container Specifications

### 1. Agent Orchestrator Service (`agent-orchestrator`)

**Base Image**: `python:3.10-slim`

**Ports**:
- `8770:8770` (Orchestration API)
- `8771:8771` (WebSocket Events)

**Volumes**:
- `./services/agent-orchestrator:/app`
- `./data/agents:/app/agents`
- `./configs/agent-templates:/app/templates`

**Environment Variables**:
```env
DATABASE_URL=postgresql://user:pass@postgres:5432/agents
REDIS_URL=redis://redis:6379
AUTHORIZATION_SERVICE_URL=http://authorization:8763
LLM_RUNTIME_URL=http://llm-runtime:8772
RAG_ENGINE_URL=http://rag-engine:8773
VECTOR_DB_URL=http://vector-db:8774
AGENT_TIMEOUT=3600
MAX_CONCURRENT_AGENTS=10
MODEL_CACHE_SIZE=50GB
```

### 2. LLM Runtime Service (`llm-runtime`)

**Base Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

**Ports**:
- `8772:8772` (LLM API)

**GPU Requirements**:
- NVIDIA GPU with 24GB+ VRAM (for large models)
- CUDA 12.1+ support

**Volumes**:
- `./services/llm-runtime:/app`
- `./data/models:/app/models`
- `./data/lora-adaptors:/app/adaptors`
- `./data/cache:/app/cache`

**Environment Variables**:
```env
CUDA_VISIBLE_DEVICES=all
MODEL_CACHE_DIR=/app/models
ADAPTOR_CACHE_DIR=/app/adaptors
BASE_MODELS=llama-2-7b,llama-2-13b,codellama-7b,mistral-7b
MAX_SEQUENCE_LENGTH=4096
BATCH_SIZE=8
INFERENCE_ENGINE=vllm
QUANTIZATION=4bit
```

### 3. RAG Engine Service (`rag-engine`)

**Base Image**: `python:3.10-slim`

**Ports**:
- `8773:8773` (RAG API)

**Volumes**:
- `./services/rag-engine:/app`
- `./data/embeddings:/app/embeddings`
- `./data/documents:/app/documents`

**Environment Variables**:
```env
VECTOR_DB_URL=http://vector-db:8774
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
RETRIEVAL_TOP_K=10
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
CHUNK_OVERLAP_RATIO=0.1
MAX_CONTEXT_LENGTH=8192
SIMILARITY_THRESHOLD=0.7
```

### 4. Vector Database Service (`vector-db`)

**Base Image**: `qdrant/qdrant:v1.7.0`

**Ports**:
- `8774:6333` (Qdrant API)
- `8775:6334` (Qdrant gRPC)

**Volumes**:
- `./data/vector-storage:/qdrant/storage`
- `./configs/qdrant:/qdrant/config`

**Environment Variables**:
```env
QDRANT__SERVICE__HTTP_PORT=6333
QDRANT__SERVICE__GRPC_PORT=6334
QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=128
QDRANT__SERVICE__MAX_WORKERS=8
QDRANT__CLUSTER__ENABLED=false
```

### 5. Agent Manager Service (`agent-manager`)

**Base Image**: `node:18-alpine`

**Ports**:
- `8776:8776` (Management API)
- `8777:8777` (Health Monitoring)

**Volumes**:
- `./services/agent-manager:/app`
- `./data/agent-instances:/app/instances`
- `./logs/agents:/app/logs`

**Environment Variables**:
```env
AGENT_ORCHESTRATOR_URL=http://agent-orchestrator:8770
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
LOG_LEVEL=info
HEALTH_CHECK_INTERVAL=30
SCALING_THRESHOLD_CPU=80
SCALING_THRESHOLD_MEMORY=85
```

## Agent Template System

### Agent Configuration Schema

```yaml
apiVersion: deaap.io/v1
kind: AgentTemplate
metadata:
  name: openbmc-expert
  domain: hardware
  version: "1.0.0"
  description: "OpenBMC hardware management expert agent"
spec:
  baseModel:
    name: "llama-2-7b-chat"
    quantization: "4bit"
  loraAdaptors:
    - name: "openbmc-code-adaptor"
      weight: 1.0
      required: true
      authorization:
        required: true
        validators: ["hardware-bu", "security-bu"]
    - name: "hardware-troubleshooting-adaptor"
      weight: 0.8
      required: false
      authorization:
        required: true
        validators: ["hardware-bu"]
  ragConfig:
    collections:
      - name: "openbmc-documentation"
        weight: 1.0
        authorization:
          required: true
          validators: ["hardware-bu", "documentation-bu"]
      - name: "hardware-specifications"
        weight: 0.9
        authorization:
          required: true
          validators: ["hardware-bu"]
    retrievalConfig:
      topK: 5
      rerankingEnabled: true
      contextWindow: 4096
  systemPrompt: |
    You are an OpenBMC expert assistant specializing in hardware management,
    firmware development, and system troubleshooting. You have deep knowledge
    of BMC architectures, IPMI protocols, and hardware monitoring systems.
  capabilities:
    - code_generation
    - troubleshooting
    - documentation_search
    - hardware_analysis
  constraints:
    maxTokens: 2048
    temperature: 0.7
    timeout: 300
  authorization:
    requiredValidators: ["hardware-bu", "security-bu"]
    consensusThreshold: 0.75
    validationTimeout: 300
```

### Domain-Specific Agent Examples

#### Financial Analysis Agent

```yaml
apiVersion: deaap.io/v1
kind: AgentTemplate
metadata:
  name: financial-analyst
  domain: finance
spec:
  baseModel:
    name: "llama-2-13b-chat"
  loraAdaptors:
    - name: "financial-analysis-adaptor"
      authorization:
        validators: ["finance-bu", "compliance-bu", "legal-bu"]
  ragConfig:
    collections:
      - name: "financial-reports"
        authorization:
          validators: ["finance-bu", "compliance-bu"]
      - name: "market-data"
        authorization:
          validators: ["finance-bu"]
  authorization:
    requiredValidators: ["finance-bu", "compliance-bu", "legal-bu"]
    consensusThreshold: 1.0  # Unanimous approval for financial data
```

#### Legal Research Agent

```yaml
apiVersion: deaap.io/v1
kind: AgentTemplate
metadata:
  name: legal-researcher
  domain: legal
spec:
  baseModel:
    name: "codellama-7b-instruct"
  loraAdaptors:
    - name: "legal-research-adaptor"
      authorization:
        validators: ["legal-bu", "compliance-bu"]
  ragConfig:
    collections:
      - name: "case-law"
        authorization:
          validators: ["legal-bu"]
      - name: "regulations"
        authorization:
          validators: ["legal-bu", "compliance-bu"]
  authorization:
    requiredValidators: ["legal-bu", "compliance-bu"]
    consensusThreshold: 0.8
```

## Agent Lifecycle Management

### Agent Creation Flow

```mermaid
sequenceDiagram
    participant User
    participant AO as Agent Orchestrator
    participant AS as Authorization Service
    participant LM as LORA Manager
    participant EM as Embedding Manager
    participant LLM as LLM Runtime
    participant RAG as RAG Engine

    User->>AO: Request agent creation
    AO->>AS: Check authorization for assets
    AS->>AS: Validate consensus results
    AS->>AO: Return authorized assets
    AO->>LM: Load authorized LORA adaptors
    AO->>EM: Load authorized embeddings
    LM->>LLM: Configure model with adaptors
    EM->>RAG: Configure RAG with embeddings
    AO->>User: Agent ready for use
```

### Runtime Management

1. **Session Management**
   - Unique session IDs for each user interaction
   - Session-based context management
   - Automatic session cleanup after timeout

2. **Resource Allocation**
   - Dynamic GPU memory allocation
   - CPU/Memory limits per agent instance
   - Queue management for high demand

3. **Performance Monitoring**
   - Response time tracking
   - Token usage monitoring
   - Resource utilization metrics

## Data Flow Specification

### Agent Request Processing

```python
class AgentRequestProcessor:
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        # 1. Validate user permissions
        user_auth = await self.authorize_user(request.user_id, request.domain)
        
        # 2. Load agent template
        template = await self.load_agent_template(request.agent_type)
        
        # 3. Check asset authorizations
        authorized_assets = await self.check_asset_authorizations(
            template.required_assets, request.user_id
        )
        
        # 4. Instantiate agent with authorized assets
        agent = await self.create_agent_instance(template, authorized_assets)
        
        # 5. Process user query
        response = await agent.process_query(request.query, request.context)
        
        # 6. Log interaction for audit
        await self.log_interaction(request, response, authorized_assets)
        
        return response
```

### Asset Authorization Check

```python
class AssetAuthorizationChecker:
    async def check_authorization(self, asset_id: str, user_id: str) -> AuthResult:
        # 1. Get asset metadata
        asset = await self.get_asset_metadata(asset_id)
        
        # 2. Check consensus results
        consensus = await self.get_consensus_result(asset.consensus_request_id)
        
        # 3. Verify consensus validity
        if not self.is_consensus_valid(consensus):
            return AuthResult(authorized=False, reason="Invalid consensus")
        
        # 4. Check user permissions
        user_permission = await self.check_user_permission(user_id, asset_id)
        
        # 5. Apply time-based restrictions
        if not self.check_time_restrictions(asset, user_id):
            return AuthResult(authorized=False, reason="Time restriction")
        
        return AuthResult(
            authorized=True,
            expires_at=self.calculate_expiration(asset, user_id),
            conditions=self.get_usage_conditions(asset, user_id)
        )
```

## API Specifications

### Agent Orchestrator API

**POST** `/agents/create`

```json
{
  "template": "openbmc-expert",
  "user_id": "user123",
  "session_config": {
    "timeout": 3600,
    "max_tokens": 2048,
    "context_window": 4096
  },
  "preferences": {
    "response_style": "detailed",
    "include_sources": true
  }
}
```

**Response**:

```json
{
  "agent_id": "agent_openbmc_20241128_123456",
  "session_id": "sess_789abc",
  "status": "ready",
  "authorized_assets": {
    "lora_adaptors": ["openbmc-code-adaptor"],
    "embeddings": ["openbmc-documentation", "hardware-specifications"]
  },
  "capabilities": ["code_generation", "troubleshooting"],
  "expires_at": "2024-11-28T23:59:59Z"
}
```

**POST** `/agents/{agent_id}/query`

```json
{
  "query": "How do I configure IPMI settings for fan control?",
  "context": {
    "hardware_type": "server",
    "bmc_version": "2.15.0"
  },
  "options": {
    "include_code": true,
    "max_tokens": 1024
  }
}
```

**Response**:

```json
{
  "response": "To configure IPMI fan control settings...",
  "sources": [
    {
      "type": "documentation",
      "title": "OpenBMC Fan Control Guide",
      "relevance": 0.95
    }
  ],
  "code_examples": ["ipmi_fan_config.c"],
  "tokens_used": 856,
  "response_time_ms": 1247
}
```

## Performance & Scalability

### Performance Targets

- **Agent Creation**: < 30 seconds
- **Query Response**: < 5 seconds (95th percentile)
- **Concurrent Agents**: 100+ per GPU node
- **Throughput**: 1000+ queries/minute per node

### Scaling Strategies

1. **Horizontal Scaling**
   - Multiple LLM runtime instances
   - Load balancing across GPU nodes
   - Distributed vector database

2. **Resource Optimization**
   - Model quantization (4-bit, 8-bit)
   - Dynamic batching
   - Cache optimization

3. **Auto-scaling**
   - Kubernetes-based scaling
   - GPU resource pooling
   - Demand-based provisioning

## Security & Compliance

### Access Control

- **Authentication**: JWT tokens with expiration
- **Authorization**: Role-based access control (RBAC)
- **Session Management**: Secure session tokens
- **Rate Limiting**: Per-user and per-endpoint limits

### Data Protection

- **In-Transit**: TLS 1.3 for all communications
- **At-Rest**: AES-256 encryption for stored data
- **Memory**: Secure memory clearing after use
- **Audit Trail**: Complete interaction logging

### Compliance

- **GDPR**: Data minimization and right to deletion
- **SOC2**: Security controls and monitoring
- **ISO 27001**: Information security management
- **Industry-Specific**: Finance (SOX), Healthcare (HIPAA)
