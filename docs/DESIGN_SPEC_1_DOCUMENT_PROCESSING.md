# System Design Specification 1: Document Processing Pipeline

## Overview

This specification defines the document processing pipeline that transforms raw PDF documents into structured, validated data assets for LLM agent construction.

## Component Architecture

```mermaid
graph TD
    subgraph "External Input"
        PDF[PDF Documents]
        API[API Gateway]
        WEB[Web Interface]
    end
    
    subgraph "Document Ingestion Service"
        DIS[Doc Ingester API]
        TM[Task Manager]
        QM[Queue Manager]
        PDF --> DIS
        API --> DIS
        WEB --> DIS
        DIS --> TM
        TM --> QM
    end
    
    subgraph "Processing Engines"
        ME[Mineru Engine]
        YYC[YY-Chunkers]
        IMG[Image Processor]
        QM --> ME
        ME --> YYC
        ME --> IMG
    end
    
    subgraph "Data Pipeline"
        SDG[Synthetic Dataset Generator]
        EMB[Embedding Generator]
        META[Metadata Extractor]
        YYC --> SDG
        YYC --> EMB
        YYC --> META
        IMG --> EMB
    end
    
    subgraph "Storage Layer"
        FS[File Storage]
        CACHE[Redis Cache]
        TDB[Task Database]
        SDG --> FS
        EMB --> FS
        META --> TDB
        TM --> CACHE
    end
    
    class DIS,TM,QM fill:#e1f5fe
    class ME,YYC,IMG fill:#f3e5f5
    class SDG,EMB,META fill:#e8f5e8
    class FS,CACHE,TDB fill:#fff3e0
```

## Container Specifications

### 1. Document Ingester Service (`doc-ingester`)

**Base Image**: `python:3.10-slim`

**Ports**: 
- `8752:8752` (FastAPI service)

**Volumes**:
- `./services/doc-ingester:/app`
- `./data/uploads:/app/uploads`
- `./data/results:/app/results`

**Environment Variables**:
```env
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://user:pass@postgres:5432/deaap
MINERU_SERVICE_URL=http://mineru:8753
YY_CHUNKER_URL=http://yy-chunker:8754
MAX_FILE_SIZE=100MB
SUPPORTED_FORMATS=pdf,docx,txt
```

**Dependencies**:
- Redis (caching and task queue)
- PostgreSQL (metadata storage)
- Mineru service
- YY-Chunker service

### 2. Mineru Engine Service (`mineru`)

**Base Image**: `registry.inventec/proxy/pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`

**Ports**:
- `8753:8753` (Processing API)

**GPU Requirements**:
- NVIDIA GPU with CUDA support
- Minimum 8GB VRAM

**Volumes**:
- `./services/mineru:/app`
- `./data/models:/app/models`
- `./data/temp:/app/temp`

**Environment Variables**:
```env
CUDA_VISIBLE_DEVICES=all
MODEL_CACHE_DIR=/app/models
TEMP_DIR=/app/temp
MAX_PAGES_PER_REQUEST=50
OUTPUT_FORMAT=markdown
```

### 3. YY-Chunker Service (`yy-chunker`)

**Base Image**: `python:3.10-slim`

**Ports**:
- `8754:8754` (Chunking API)

**Volumes**:
- `./services/yy-chunker:/app`
- `./data/chunks:/app/chunks`

**Environment Variables**:
```env
DEFAULT_CHUNK_SIZE=2100
DEFAULT_OVERLAP=200
CHUNKER_METHODS=semantic,fixed_token,recursive,markdown
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 4. Synthetic Dataset Generator (`synthetic-data`)

**Base Image**: `python:3.10-slim`

**Ports**:
- `8755:8755` (Generation API)

**Volumes**:
- `./services/synthetic-data:/app`
- `./data/datasets:/app/datasets`

**Environment Variables**:
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
DATASET_FORMATS=alpaca,sharegpt,instruction
SYNTHETIC_RATIO=3
QUALITY_THRESHOLD=0.8
```

### 5. Embedding Generator (`embedding-service`)

**Base Image**: `python:3.10-slim`

**Ports**:
- `8756:8756` (Embedding API)

**GPU Requirements** (Optional):
- NVIDIA GPU for faster embedding generation

**Volumes**:
- `./services/embedding-service:/app`
- `./data/embeddings:/app/embeddings`
- `./data/models:/app/models`

**Environment Variables**:
```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
BATCH_SIZE=32
VECTOR_DIMENSION=768
SIMILARITY_THRESHOLD=0.7
```

## Data Flow Specification

### Input Data Flow

1. **Document Upload**
   ```
   User/API → Doc Ingester → Task Queue → Mineru Engine
   ```

2. **PDF Processing**
   ```
   Mineru Engine → Markdown + Images → YY-Chunker
   ```

3. **Chunking**
   ```
   YY-Chunker → Multiple Chunk Strategies → Metadata Extraction
   ```

### Processing Data Flow

1. **Synthetic Data Generation**
   ```
   Chunks → LLM Provider → Synthetic QA Pairs → Dataset Files
   ```

2. **Embedding Generation**
   ```
   Chunks + Images → Embedding Model → Vector Embeddings → Vector Store
   ```

3. **Metadata Processing**
   ```
   All Outputs → Metadata Extraction → Structured Metadata → Database
   ```

## API Specifications

### Document Ingester API

**POST** `/transform`
```json
{
  "url": "string",
  "start_page": 1,
  "end_page": 100,
  "chunk_method": ["CLUSTER_SEMANTIC", "FIXED_TOKEN"],
  "chunk_max_size": 5100,
  "chunk_overlap": 200,
  "generate_synthetic": true,
  "generate_embeddings": true
}
```

**GET** `/status/{task_id}`
```json
{
  "state": true,
  "status": "processing|completed|failed",
  "progress": 85,
  "current_step": "embedding_generation",
  "results": {
    "chunks_count": 150,
    "synthetic_pairs": 450,
    "embeddings_count": 150,
    "metadata": {...}
  }
}
```

### Synthetic Data API

**POST** `/generate`
```json
{
  "chunks": ["chunk1", "chunk2"],
  "format": "alpaca",
  "count_multiplier": 3,
  "quality_filter": true
}
```

### Embedding API

**POST** `/embed`
```json
{
  "texts": ["text1", "text2"],
  "model": "all-mpnet-base-v2",
  "batch_size": 32
}
```

## Performance Requirements

### Throughput
- **Document Processing**: 10 PDFs/hour (100-page average)
- **Chunking**: 1000 chunks/minute
- **Synthetic Generation**: 100 QA pairs/minute
- **Embedding Generation**: 500 embeddings/minute

### Latency
- **API Response**: < 200ms
- **Task Status**: < 100ms
- **Processing Start**: < 5 seconds

### Resource Utilization
- **CPU**: < 80% average utilization
- **Memory**: < 16GB per service
- **GPU**: < 90% utilization during processing
- **Storage**: 1TB minimum for data storage

## Error Handling & Recovery

### Retry Mechanisms
- **Network Failures**: 3 retries with exponential backoff
- **Processing Failures**: 2 retries with different parameters
- **Resource Constraints**: Queue with priority handling

### Data Consistency
- **Atomic Operations**: All-or-nothing processing
- **Checkpoint System**: Resume from last successful step
- **Validation**: Schema validation at each stage

### Monitoring & Alerting
- **Health Checks**: Every 30 seconds
- **Performance Metrics**: CPU, memory, GPU utilization
- **Business Metrics**: Processing success rate, throughput
- **Alert Conditions**: Service down, high error rate, resource exhaustion

## Security Considerations

### Data Protection
- **Encryption**: AES-256 for data at rest
- **Network**: TLS 1.3 for all communications
- **Access Control**: JWT tokens with role-based permissions

### Input Validation
- **File Types**: Whitelist of allowed formats
- **File Size**: Configurable limits per upload
- **Content Scanning**: Malware detection
- **Rate Limiting**: Per-user and per-IP limits

### Audit Trail
- **Request Logging**: All API calls with timestamps
- **Processing Logs**: Step-by-step execution logs
- **Data Lineage**: Track data transformation chain
- **Compliance**: GDPR, SOC2 compliance ready
