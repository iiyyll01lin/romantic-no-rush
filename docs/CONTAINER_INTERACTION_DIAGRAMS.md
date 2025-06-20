# DEAAP Container Interaction Diagrams

## Overview

This document provides detailed container interaction diagrams for key subsystems within the Decentralized Enterprise AI Agent Platform (DEAAP). These diagrams complement the comprehensive microservices documentation and design specifications already established.

## Table of Contents

1. [Document Processing Subsystem](#document-processing-subsystem)
2. [Consensus Validation Subsystem](#consensus-validation-subsystem)
3. [AI Agent Orchestration Subsystem](#ai-agent-orchestration-subsystem)
4. [Security & Privacy Subsystem](#security--privacy-subsystem)
5. [Data Storage & Retrieval Subsystem](#data-storage--retrieval-subsystem)
6. [Monitoring & Observability Subsystem](#monitoring--observability-subsystem)
7. [Network Architecture Overview](#network-architecture-overview)

---

## Document Processing Subsystem

### Container Interaction Flow

```mermaid
graph TB
    subgraph "External Input Layer"
        API[API Gateway<br/>Port: 8750<br/>Network: deaap-api]
        WEB[Web Interface<br/>Port: 3000<br/>Network: deaap-api]
    end
    
    subgraph "Document Processing Network (deaap-processing)"
        DI[Doc Ingester<br/>Port: 8752<br/>CPU: 2 cores<br/>Memory: 4GB]
        
        subgraph "Processing Engines"
            MU[MinerU Engine<br/>Port: 8753<br/>GPU: Required<br/>Memory: 8GB<br/>Runtime: nvidia]
            YY[YY-Chunker<br/>Port: 8754<br/>CPU: 4 cores<br/>Memory: 6GB]
            IMG[Image Processor<br/>Port: 8755<br/>GPU: Optional<br/>Memory: 4GB]
        end
        
        subgraph "Data Generation"
            SG[Synthetic Generator<br/>Port: 8756<br/>CPU: 2 cores<br/>Memory: 4GB]
            EG[Embedding Generator<br/>Port: 8757<br/>GPU: Required<br/>Memory: 6GB]
            LG[LORA Generator<br/>Port: 8758<br/>CPU: 4 cores<br/>Memory: 8GB]
        end
    end
    
    subgraph "Data Storage Network (deaap-storage)"
        RD[Redis Cache<br/>Port: 6379<br/>Memory: 2GB]
        PG[PostgreSQL<br/>Port: 5432<br/>Memory: 4GB<br/>Storage: 100GB]
        FS[File Storage<br/>NFS Mount<br/>Storage: 1TB]
        VD[Vector DB<br/>Port: 8200<br/>Memory: 8GB<br/>Storage: 500GB]
    end
    
    subgraph "Message Queue Network (deaap-messaging)"
        MQ[RabbitMQ<br/>Port: 5672<br/>Management: 15672<br/>Memory: 2GB]
    end

    %% External connections
    API --> DI
    WEB --> DI
    
    %% Processing flow
    DI --> MU
    DI --> RD
    DI --> MQ
    
    MU --> YY
    MU --> IMG
    
    YY --> SG
    YY --> EG
    YY --> LG
    
    %% Data persistence
    DI --> PG
    SG --> FS
    EG --> VD
    LG --> FS
    
    %% Queue management
    DI --> MQ
    MU --> MQ
    YY --> MQ
    
    %% Health checks and monitoring
    DI -.->|Health Check| MU
    DI -.->|Health Check| YY
    DI -.->|Health Check| IMG
    
    class API,WEB fill:#e1f5fe
    class DI fill:#f3e5f5
    class MU,YY,IMG fill:#e8f5e8
    class SG,EG,LG fill:#fff3e0
    class RD,PG,FS,VD fill:#fce4ec
    class MQ fill:#f1f8e9
```

### Container Specifications

| Container | Image | CPU | Memory | Storage | Ports | Networks |
|-----------|-------|-----|--------|---------|-------|----------|
| doc-ingester | deaap/doc-ingester:latest | 2 cores | 4GB | 10GB | 8752:8752 | deaap-processing, deaap-storage |
| mineru | deaap/mineru:v1.3.3 | 4 cores | 8GB | 20GB | 8753:8753 | deaap-processing |
| yy-chunker | deaap/yy-chunker:latest | 4 cores | 6GB | 5GB | 8754:8754 | deaap-processing |
| synthetic-gen | deaap/synthetic-gen:latest | 2 cores | 4GB | 5GB | 8756:8756 | deaap-processing |
| embedding-gen | deaap/embedding-gen:latest | 2 cores | 6GB | 5GB | 8757:8757 | deaap-processing |
| lora-gen | deaap/lora-gen:latest | 4 cores | 8GB | 10GB | 8758:8758 | deaap-processing |

---

## Consensus Validation Subsystem

### Consensus Container Architecture

```mermaid
graph TB
    subgraph "Consensus Management Network (deaap-consensus)"
        CM[Consensus Manager<br/>Port: 8760<br/>WebSocket: 8761<br/>CPU: 4 cores<br/>Memory: 6GB]
        
        subgraph "BU Validator Nodes"
            BU1[BU-1 Validator<br/>Port: 8765<br/>CPU: 2 cores<br/>Memory: 4GB]
            BU2[BU-2 Validator<br/>Port: 8766<br/>CPU: 2 cores<br/>Memory: 4GB]
            BU3[BU-3 Validator<br/>Port: 8767<br/>CPU: 2 cores<br/>Memory: 4GB]
            BUN[BU-N Validator<br/>Port: 8768<br/>CPU: 2 cores<br/>Memory: 4GB]
        end
        
        subgraph "Cartesi Infrastructure"
            CR[Cartesi Runtime<br/>Port: 5004<br/>CPU: 4 cores<br/>Memory: 8GB]
            CM_PROXY[Cartesi Machine Proxy<br/>Port: 5005<br/>CPU: 2 cores<br/>Memory: 4GB]
        end
    end
    
    subgraph "Blockchain Network (deaap-blockchain)"
        BN[Blockchain Node<br/>Port: 8545<br/>RPC: 8546<br/>CPU: 4 cores<br/>Memory: 8GB]
        SC[Smart Contracts<br/>Consensus Logic<br/>Data Authorization]
    end
    
    subgraph "Storage & Cache Network (deaap-storage)"
        RD2[Redis Cache<br/>Port: 6379<br/>Memory: 4GB]
        PG2[PostgreSQL<br/>Port: 5432<br/>Memory: 6GB<br/>Storage: 200GB]
    end
    
    subgraph "External Integration"
        DI2[Doc Ingester<br/>Validation Requests]
        AO[Agent Orchestrator<br/>Authorization Checks]
    end

    %% Validation flow
    DI2 --> CM
    CM --> BU1
    CM --> BU2
    CM --> BU3
    CM --> BUN
    
    %% Cartesi execution
    BU1 --> CR
    BU2 --> CR
    BU3 --> CR
    BUN --> CR
    
    CR --> CM_PROXY
    CM_PROXY --> CM
    
    %% Blockchain integration
    CM --> BN
    BN --> SC
    SC --> BN
    BN --> CM
    
    %% Data persistence
    CM --> RD2
    CM --> PG2
    BU1 --> RD2
    BU2 --> RD2
    BU3 --> RD2
    BUN --> RD2
    
    %% Authorization
    CM --> AO
    
    %% WebSocket connections for real-time updates
    CM -.->|WebSocket| BU1
    CM -.->|WebSocket| BU2
    CM -.->|WebSocket| BU3
    CM -.->|WebSocket| BUN
    
    class DI2,AO fill:#e1f5fe
    class CM fill:#f3e5f5
    class BU1,BU2,BU3,BUN fill:#e8f5e8
    class CR,CM_PROXY fill:#fff3e0
    class BN,SC fill:#fce4ec
    class RD2,PG2 fill:#f1f8e9
```

### Consensus Protocol Sequence

```mermaid
sequenceDiagram
    participant DI as Doc Ingester
    participant CM as Consensus Manager
    participant BU1 as BU-1 Validator
    participant BU2 as BU-2 Validator
    participant BU3 as BU-3 Validator
    participant CR as Cartesi Runtime
    participant BN as Blockchain Node

    DI->>CM: Submit validation request
    CM->>CM: Create validation session
    
    par Parallel Validation
        CM->>BU1: Assign validation task
        CM->>BU2: Assign validation task
        CM->>BU3: Assign validation task
    end
    
    par Cartesi Execution
        BU1->>CR: Execute validation logic
        BU2->>CR: Execute validation logic
        BU3->>CR: Execute validation logic
    end
    
    par Submit Results
        CR->>BU1: Return validation result
        CR->>BU2: Return validation result
        CR->>BU3: Return validation result
    end
    
    par Report to Consensus Manager
        BU1->>CM: Submit validation result
        BU2->>CM: Submit validation result
        BU3->>CM: Submit validation result
    end
    
    CM->>CM: Aggregate results (threshold: 67%)
    CM->>BN: Submit consensus result
    BN->>CM: Confirmation & receipt
    CM->>DI: Validation complete
```

---

## AI Agent Orchestration Subsystem

### Agent Orchestration Architecture

```mermaid
graph TB
    subgraph "Agent Orchestration Network (deaap-agent)"
        AO[Agent Orchestrator<br/>Port: 8770<br/>WebSocket: 8771<br/>CPU: 4 cores<br/>Memory: 8GB]
        
        subgraph "LLM Runtime Cluster"
            LR1[LLM Runtime-1<br/>Port: 8775<br/>GPU: Required<br/>Memory: 16GB]
            LR2[LLM Runtime-2<br/>Port: 8776<br/>GPU: Required<br/>Memory: 16GB]
            LRN[LLM Runtime-N<br/>Port: 8777<br/>GPU: Required<br/>Memory: 16GB]
        end
        
        subgraph "Model Management"
            MR[Model Registry<br/>Port: 8780<br/>CPU: 2 cores<br/>Memory: 4GB<br/>Storage: 500GB]
            LA[LORA Adapter Hub<br/>Port: 8781<br/>CPU: 2 cores<br/>Memory: 4GB<br/>Storage: 200GB]
        end
        
        subgraph "RAG System"
            RAG[RAG Engine<br/>Port: 8785<br/>CPU: 4 cores<br/>Memory: 8GB]
            ES[Elasticsearch<br/>Port: 9200<br/>Memory: 8GB<br/>Storage: 1TB]
        end
    end
    
    subgraph "Data Access Network (deaap-storage)"
        VD2[Vector Database<br/>Port: 8200<br/>Memory: 12GB]
        FS2[File Storage<br/>LORA & Embeddings<br/>Storage: 2TB]
        RD3[Redis Cache<br/>Model Cache<br/>Memory: 8GB]
    end
    
    subgraph "External Services"
        CM2[Consensus Manager<br/>Authorization]
        DI3[Doc Ingester<br/>Data Access]
        API2[API Gateway<br/>Client Requests]
    end

    %% External requests
    API2 --> AO
    CM2 --> AO
    
    %% Orchestration flow
    AO --> MR
    AO --> LA
    AO --> RAG
    
    %% LLM distribution
    AO --> LR1
    AO --> LR2
    AO --> LRN
    
    %% Model loading
    MR --> LR1
    MR --> LR2
    MR --> LRN
    
    LA --> LR1
    LA --> LR2
    LA --> LRN
    
    %% RAG operations
    RAG --> ES
    RAG --> VD2
    RAG --> LR1
    RAG --> LR2
    RAG --> LRN
    
    %% Data access
    AO --> VD2
    AO --> FS2
    AO --> RD3
    
    MR --> FS2
    LA --> FS2
    
    %% Caching
    LR1 --> RD3
    LR2 --> RD3
    LRN --> RD3
    
    %% Data source
    DI3 --> VD2
    DI3 --> FS2
    
    %% Load balancing indicators
    AO -.->|Load Balance| LR1
    AO -.->|Load Balance| LR2
    AO -.->|Load Balance| LRN
    
    class API2,CM2,DI3 fill:#e1f5fe
    class AO fill:#f3e5f5
    class LR1,LR2,LRN fill:#e8f5e8
    class MR,LA fill:#fff3e0
    class RAG,ES fill:#fce4ec
    class VD2,FS2,RD3 fill:#f1f8e9
```

### Agent Lifecycle Management

```mermaid
stateDiagram-v2
    [*] --> Requested
    Requested --> Validating: Authorization Check
    Validating --> Preparing: Consensus Approved
    Validating --> Rejected: Consensus Denied
    
    Preparing --> Loading: Model Selection
    Loading --> Configuring: LORA Adaptation
    Configuring --> Ready: RAG Integration
    
    Ready --> Running: Client Request
    Running --> Processing: LLM Inference
    Processing --> Responding: Generate Response
    Responding --> Ready: Response Delivered
    
    Running --> Paused: Resource Constraint
    Paused --> Ready: Resources Available
    
    Ready --> Scaling: High Load
    Scaling --> Ready: Auto-scaling Complete
    
    Ready --> Updating: Model Update
    Updating --> Ready: Update Complete
    
    Ready --> Terminating: Session End
    Terminating --> [*]
    
    Rejected --> [*]
```

---

## Security & Privacy Subsystem

### Security Infrastructure Architecture

```mermaid
graph TB
    subgraph "Security Gateway Network (deaap-security)"
        SG[Security Gateway<br/>Port: 8800<br/>TLS Termination<br/>CPU: 4 cores<br/>Memory: 4GB]
        
        subgraph "Identity & Access Management"
            KC[Keycloak IAM<br/>Port: 8080<br/>CPU: 2 cores<br/>Memory: 4GB]
            AU[Auth Service<br/>Port: 8801<br/>CPU: 2 cores<br/>Memory: 2GB]
            RL[Rate Limiter<br/>Port: 8802<br/>CPU: 1 core<br/>Memory: 1GB]
        end
        
        subgraph "Encryption & Privacy"
            EG[Encryption Gateway<br/>Port: 8810<br/>HSM Integration<br/>CPU: 2 cores<br/>Memory: 2GB]
            DLP[DLP Service<br/>Port: 8811<br/>CPU: 2 cores<br/>Memory: 4GB]
            PP[Privacy Processor<br/>Port: 8812<br/>CPU: 2 cores<br/>Memory: 4GB]
        end
        
        subgraph "Monitoring & Audit"
            AS[Audit Service<br/>Port: 8820<br/>CPU: 2 cores<br/>Memory: 4GB]
            TM[Threat Monitor<br/>Port: 8821<br/>CPU: 2 cores<br/>Memory: 4GB]
            IR[Incident Response<br/>Port: 8822<br/>CPU: 1 core<br/>Memory: 2GB]
        end
    end
    
    subgraph "Secure Storage Network (deaap-secure-storage)"
        HSM[Hardware Security Module<br/>Key Management<br/>Network: Isolated]
        ADB[Audit Database<br/>Port: 5433<br/>Encrypted Storage<br/>Memory: 6GB]
        KV[Key-Value Store<br/>Port: 8500<br/>Consul/Vault<br/>Memory: 2GB]
    end
    
    subgraph "Application Networks"
        APP1[Doc Processing<br/>deaap-processing]
        APP2[Consensus<br/>deaap-consensus]
        APP3[Agent Orchestration<br/>deaap-agent]
    end
    
    subgraph "External Interfaces"
        EXT[External Clients<br/>HTTPS/WSS]
        COMP[Compliance Systems<br/>SIEM/SOC]
    end

    %% External traffic flow
    EXT --> SG
    SG --> KC
    SG --> AU
    SG --> RL
    
    %% Security processing
    SG --> EG
    EG --> DLP
    DLP --> PP
    
    %% Application access
    SG --> APP1
    SG --> APP2
    SG --> APP3
    
    %% Security enforcement
    AU --> APP1
    AU --> APP2
    AU --> APP3
    
    %% Encryption services
    EG --> HSM
    EG --> KV
    
    %% Monitoring & audit
    SG --> AS
    AU --> AS
    EG --> AS
    
    AS --> ADB
    AS --> TM
    TM --> IR
    
    %% Compliance reporting
    AS --> COMP
    TM --> COMP
    
    %% Health monitoring
    TM -.->|Monitor| APP1
    TM -.->|Monitor| APP2
    TM -.->|Monitor| APP3
    
    class EXT,COMP fill:#e1f5fe
    class SG fill:#f3e5f5
    class KC,AU,RL fill:#e8f5e8
    class EG,DLP,PP fill:#fff3e0
    class AS,TM,IR fill:#fce4ec
    class HSM,ADB,KV fill:#f1f8e9
    class APP1,APP2,APP3 fill:#e8eaf6
```

### Security Flow Sequence

```mermaid
sequenceDiagram
    participant Client
    participant SG as Security Gateway
    participant KC as Keycloak
    participant AU as Auth Service
    participant EG as Encryption Gateway
    participant DLP as DLP Service
    participant APP as Application
    participant AS as Audit Service

    Client->>SG: HTTPS Request
    SG->>KC: Authenticate User
    KC->>AU: Validate Token
    AU->>SG: Authorization Result
    
    alt Authorized
        SG->>EG: Encrypt Payload
        EG->>DLP: Scan for PII
        DLP->>EG: DLP Result
        EG->>SG: Processed Payload
        SG->>APP: Forward Request
        APP->>SG: Response
        SG->>EG: Encrypt Response
        EG->>SG: Encrypted Response
        SG->>Client: HTTPS Response
        
        par Audit Trail
            SG->>AS: Log Request
            AU->>AS: Log Authorization
            EG->>AS: Log Encryption
            DLP->>AS: Log DLP Scan
            APP->>AS: Log Processing
        end
    else Unauthorized
        SG->>AS: Log Unauthorized Access
        SG->>Client: 401 Unauthorized
    end
```

---

## Data Storage & Retrieval Subsystem

### Storage Cluster Architecture

```mermaid
graph TB
    subgraph "Storage Management Network (deaap-storage)"
        SM[Storage Manager<br/>Port: 8900<br/>CPU: 2 cores<br/>Memory: 4GB]
        BM[Backup Manager<br/>Port: 8901<br/>CPU: 2 cores<br/>Memory: 2GB]
        
        subgraph "Primary Storage Cluster"
            PG_M[PostgreSQL Master<br/>Port: 5432<br/>CPU: 4 cores<br/>Memory: 8GB<br/>Storage: 500GB]
            PG_S1[PostgreSQL Slave-1<br/>Port: 5433<br/>CPU: 2 cores<br/>Memory: 6GB<br/>Storage: 500GB]
            PG_S2[PostgreSQL Slave-2<br/>Port: 5434<br/>CPU: 2 cores<br/>Memory: 6GB<br/>Storage: 500GB]
        end
        
        subgraph "Vector Database Cluster"
            VD_M[Vector DB Master<br/>Port: 8200<br/>CPU: 4 cores<br/>Memory: 16GB<br/>Storage: 1TB]
            VD_S1[Vector DB Slave-1<br/>Port: 8201<br/>CPU: 2 cores<br/>Memory: 12GB<br/>Storage: 1TB]
            VD_S2[Vector DB Slave-2<br/>Port: 8202<br/>CPU: 2 cores<br/>Memory: 12GB<br/>Storage: 1TB]
        end
        
        subgraph "Cache Layer"
            RD_M[Redis Master<br/>Port: 6379<br/>CPU: 2 cores<br/>Memory: 8GB]
            RD_S1[Redis Slave-1<br/>Port: 6380<br/>CPU: 1 core<br/>Memory: 6GB]
            RD_S2[Redis Slave-2<br/>Port: 6381<br/>CPU: 1 core<br/>Memory: 6GB]
        end
        
        subgraph "File Storage"
            FS_NFS[NFS Server<br/>Port: 2049<br/>Storage: 5TB<br/>Backup: Daily]
            FS_S3[S3 Compatible<br/>Port: 9000<br/>MinIO Cluster<br/>Storage: 10TB]
        end
    end
    
    subgraph "Backup Network (deaap-backup)"
        BS[Backup Storage<br/>Off-site Replica<br/>Storage: 20TB]
        AR[Archive Service<br/>Port: 8910<br/>Long-term Storage]
    end
    
    subgraph "Application Access"
        APP_READ[Read-Heavy Apps<br/>Doc Ingester, RAG]
        APP_WRITE[Write-Heavy Apps<br/>Consensus, Audit]
        APP_CACHE[Cache Users<br/>LLM Runtime, API Gateway]
    end

    %% Storage management
    SM --> PG_M
    SM --> VD_M
    SM --> RD_M
    SM --> FS_NFS
    SM --> FS_S3
    
    %% Replication
    PG_M --> PG_S1
    PG_M --> PG_S2
    
    VD_M --> VD_S1
    VD_M --> VD_S2
    
    RD_M --> RD_S1
    RD_M --> RD_S2
    
    %% Application access patterns
    APP_READ --> PG_S1
    APP_READ --> PG_S2
    APP_READ --> VD_S1
    APP_READ --> VD_S2
    
    APP_WRITE --> PG_M
    APP_WRITE --> VD_M
    
    APP_CACHE --> RD_M
    APP_CACHE --> RD_S1
    APP_CACHE --> RD_S2
    
    %% File access
    APP_READ --> FS_NFS
    APP_READ --> FS_S3
    APP_WRITE --> FS_NFS
    APP_WRITE --> FS_S3
    
    %% Backup operations
    BM --> PG_M
    BM --> VD_M
    BM --> FS_NFS
    BM --> FS_S3
    BM --> BS
    BM --> AR
    
    %% Cross-replica sync
    PG_S1 -.->|Sync Check| PG_S2
    VD_S1 -.->|Sync Check| VD_S2
    RD_S1 -.->|Sync Check| RD_S2
    
    class SM,BM fill:#f3e5f5
    class PG_M,VD_M,RD_M fill:#e8f5e8
    class PG_S1,PG_S2,VD_S1,VD_S2,RD_S1,RD_S2 fill:#fff3e0
    class FS_NFS,FS_S3 fill:#fce4ec
    class BS,AR fill:#f1f8e9
    class APP_READ,APP_WRITE,APP_CACHE fill:#e8eaf6
```

---

## Monitoring & Observability Subsystem

### Monitoring Infrastructure Architecture

```mermaid
graph TB
    subgraph "Monitoring Network (deaap-monitoring)"
        PM[Prometheus<br/>Port: 9090<br/>Metrics Collection<br/>Memory: 4GB<br/>Storage: 200GB]
        GF[Grafana<br/>Port: 3001<br/>Dashboards<br/>Memory: 2GB]
        AM[AlertManager<br/>Port: 9093<br/>Alert Routing<br/>Memory: 1GB]
        
        subgraph "Data Collection"
            NE[Node Exporter<br/>Port: 9100<br/>Host Metrics<br/>All Nodes]
            CE[cAdvisor<br/>Port: 8080<br/>Container Metrics<br/>All Nodes]
            BE[Blackbox Exporter<br/>Port: 9115<br/>Endpoint Monitoring]
        end
        
        subgraph "Log Management"
            ELK_E[Elasticsearch<br/>Port: 9200<br/>Log Storage<br/>Memory: 8GB<br/>Storage: 1TB]
            ELK_L[Logstash<br/>Port: 5044<br/>Log Processing<br/>Memory: 4GB]
            ELK_K[Kibana<br/>Port: 5601<br/>Log Visualization<br/>Memory: 2GB]
            FB[Fluent Bit<br/>Log Collection<br/>All Containers]
        end
        
        subgraph "Tracing"
            JG[Jaeger<br/>Port: 16686<br/>Distributed Tracing<br/>Memory: 4GB]
            OT[OpenTelemetry<br/>Port: 4317<br/>Trace Collection<br/>Memory: 2GB]
        end
    end
    
    subgraph "Application Networks"
        APP_DOC[Document Processing<br/>deaap-processing]
        APP_CON[Consensus<br/>deaap-consensus]
        APP_AGT[Agent Orchestration<br/>deaap-agent]
        APP_SEC[Security<br/>deaap-security]
        APP_STO[Storage<br/>deaap-storage]
    end
    
    subgraph "External Integration"
        PD[PagerDuty<br/>Incident Management]
        SLACK[Slack<br/>Notifications]
        EMAIL[Email<br/>Alerts]
    end

    %% Metrics collection
    PM --> NE
    PM --> CE
    PM --> BE
    
    %% Application metrics
    APP_DOC --> PM
    APP_CON --> PM
    APP_AGT --> PM
    APP_SEC --> PM
    APP_STO --> PM
    
    %% Dashboards
    PM --> GF
    GF --> PM
    
    %% Alerting
    PM --> AM
    AM --> PD
    AM --> SLACK
    AM --> EMAIL
    
    %% Log pipeline
    APP_DOC --> FB
    APP_CON --> FB
    APP_AGT --> FB
    APP_SEC --> FB
    APP_STO --> FB
    
    FB --> ELK_L
    ELK_L --> ELK_E
    ELK_E --> ELK_K
    
    %% Tracing
    APP_DOC --> OT
    APP_CON --> OT
    APP_AGT --> OT
    APP_SEC --> OT
    APP_STO --> OT
    
    OT --> JG
    
    %% Cross-integration
    GF --> ELK_E
    GF --> JG
    
    %% Health monitoring
    BE -.->|Monitor| APP_DOC
    BE -.->|Monitor| APP_CON
    BE -.->|Monitor| APP_AGT
    BE -.->|Monitor| APP_SEC
    BE -.->|Monitor| APP_STO
    
    class PM,GF,AM fill:#f3e5f5
    class NE,CE,BE fill:#e8f5e8
    class ELK_E,ELK_L,ELK_K,FB fill:#fff3e0
    class JG,OT fill:#fce4ec
    class APP_DOC,APP_CON,APP_AGT,APP_SEC,APP_STO fill:#e8eaf6
    class PD,SLACK,EMAIL fill:#f1f8e9
```

### Monitoring Data Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant FB as Fluent Bit
    participant OT as OpenTelemetry
    participant PM as Prometheus
    participant ELK as ELK Stack
    participant JG as Jaeger
    participant GF as Grafana
    participant AM as AlertManager

    par Metrics Collection
        App->>PM: Metrics (scrape)
        PM->>GF: Query metrics
        PM->>AM: Evaluate rules
    and Log Collection
        App->>FB: Logs
        FB->>ELK: Process logs
        ELK->>GF: Log queries
    and Trace Collection
        App->>OT: Traces
        OT->>JG: Store traces
        JG->>GF: Trace queries
    end
    
    alt Alert Triggered
        AM->>AM: Evaluate conditions
        AM->>PD: Critical alert
        AM->>SLACK: Team notification
        AM->>EMAIL: Email alert
    end
    
    Note over GF: Unified Dashboard<br/>Metrics + Logs + Traces
```

---

## Network Architecture Overview

### Complete Network Topology

```mermaid
graph TB
    subgraph "External Access"
        INT[Internet<br/>Public Access]
        VPN[VPN Gateway<br/>Enterprise Access]
        LB[Load Balancer<br/>HA Proxy<br/>Ports: 80, 443]
    end
    
    subgraph "DMZ Network (172.20.0.0/24)"
        SG2[Security Gateway<br/>172.20.0.10]
        WAF[Web Application Firewall<br/>172.20.0.11]
        API_GW[API Gateway<br/>172.20.0.20]
    end
    
    subgraph "Application Networks"
        subgraph "Processing (172.21.0.0/24)"
            DOC[Doc Ingester<br/>172.21.0.10]
            MNU[MinerU<br/>172.21.0.20]
            YYC[YY-Chunker<br/>172.21.0.30]
        end
        
        subgraph "Consensus (172.22.0.0/24)"
            CM3[Consensus Manager<br/>172.22.0.10]
            BU[BU Validators<br/>172.22.0.20-30]
            CART[Cartesi Runtime<br/>172.22.0.40]
        end
        
        subgraph "Agent (172.23.0.0/24)"
            AO2[Agent Orchestrator<br/>172.23.0.10]
            LLM[LLM Runtime<br/>172.23.0.20-30]
            RAG2[RAG System<br/>172.23.0.40]
        end
        
        subgraph "Security (172.24.0.0/24)"
            IAM[Keycloak<br/>172.24.0.10]
            ENC[Encryption<br/>172.24.0.20]
            AUDIT[Audit<br/>172.24.0.30]
        end
        
        subgraph "Storage (172.25.0.0/24)"
            DB[Databases<br/>172.25.0.10-20]
            CACHE[Cache Layer<br/>172.25.0.30-40]
            FILES[File Storage<br/>172.25.0.50]
        end
        
        subgraph "Monitoring (172.26.0.0/24)"
            PROM[Prometheus<br/>172.26.0.10]
            GRAF[Grafana<br/>172.26.0.20]
            ELK2[ELK Stack<br/>172.26.0.30-40]
        end
    end
    
    subgraph "Management Network (172.27.0.0/24)"
        MGMT[Management<br/>172.27.0.10]
        BACKUP[Backup<br/>172.27.0.20]
        LOG[Log Archive<br/>172.27.0.30]
    end

    %% External connections
    INT --> LB
    VPN --> LB
    LB --> WAF
    WAF --> SG2
    SG2 --> API_GW
    
    %% Internal routing
    API_GW --> DOC
    API_GW --> CM3
    API_GW --> AO2
    
    %% Cross-network communication
    DOC --> CM3
    CM3 --> AO2
    
    %% Security integration
    DOC --> IAM
    CM3 --> IAM
    AO2 --> IAM
    
    %% Storage access
    DOC --> DB
    CM3 --> DB
    AO2 --> DB
    DOC --> CACHE
    CM3 --> CACHE
    AO2 --> CACHE
    
    %% Monitoring
    DOC --> PROM
    CM3 --> PROM
    AO2 --> PROM
    IAM --> PROM
    DB --> PROM
    
    %% Management
    MGMT --> DOC
    MGMT --> CM3
    MGMT --> AO2
    BACKUP --> DB
    BACKUP --> FILES
    
    class INT,VPN fill:#e1f5fe
    class LB,WAF,SG2,API_GW fill:#f3e5f5
    class DOC,MNU,YYC fill:#e8f5e8
    class CM3,BU,CART fill:#fff3e0
    class AO2,LLM,RAG2 fill:#fce4ec
    class IAM,ENC,AUDIT fill:#f1f8e9
    class DB,CACHE,FILES fill:#e8eaf6
    class PROM,GRAF,ELK2 fill:#fff8e1
    class MGMT,BACKUP,LOG fill:#f3e5f5
```

### Network Security Rules

| Source Network | Destination Network | Ports | Protocol | Purpose |
|----------------|-------------------|-------|----------|---------|
| DMZ | Processing | 8752-8758 | HTTPS | Document APIs |
| DMZ | Consensus | 8760-8770 | HTTPS/WSS | Consensus APIs |
| DMZ | Agent | 8770-8790 | HTTPS/WSS | Agent APIs |
| Processing | Consensus | 8760 | HTTPS | Validation requests |
| Consensus | Agent | 8770 | HTTPS | Authorization |
| All Apps | Security | 8800-8822 | HTTPS | Security services |
| All Apps | Storage | 5432, 6379, 8200 | TCP | Database access |
| All Apps | Monitoring | 9090-9093 | HTTP | Metrics collection |
| Management | All | 22, 3000-9999 | SSH/HTTP | Administration |

---

## Container Resource Requirements Summary

### Minimum Hardware Requirements

| Environment | CPU Cores | Memory (GB) | Storage (GB) | GPU |
|-------------|-----------|-------------|--------------|-----|
| Development | 16 | 64 | 500 | 1x RTX 3080 or better |
| Staging | 32 | 128 | 2000 | 2x RTX 4090 or better |
| Production | 64 | 256 | 5000 | 4x H100 or better |

### Container Resource Allocation

| Service Category | Containers | Total CPU | Total Memory | Total Storage |
|------------------|------------|-----------|--------------|---------------|
| Document Processing | 6 | 20 cores | 40 GB | 60 GB |
| Consensus & Blockchain | 8 | 24 cores | 44 GB | 220 GB |
| AI Agent Orchestration | 10 | 22 cores | 72 GB | 2.7 TB |
| Security & Privacy | 9 | 18 cores | 27 GB | 10 GB |
| Data Storage | 12 | 18 cores | 72 GB | 8.5 TB |
| Monitoring & Observability | 10 | 16 cores | 31 GB | 1.2 TB |
| **Total** | **55** | **118 cores** | **286 GB** | **12.5 TB** |

---

## Deployment Orchestration

### Docker Compose Profiles

```yaml
# docker-compose.yml profile structure
profiles:
  - development:  # Minimal services for local development
    - doc-ingester, mineru, yy-chunker
    - mock-consensus, mock-blockchain
    - basic-monitoring
    
  - staging:      # Full services with reduced resources
    - All processing services
    - Real consensus with 3 validators
    - Complete monitoring stack
    
  - production:   # Full services with HA
    - All services with clustering
    - Real consensus with 5+ validators
    - Complete security stack
    - Full monitoring and backup
```

### Health Check Dependencies

```mermaid
graph TB
    subgraph "Startup Order"
        L1[Layer 1: Infrastructure<br/>Networks, Volumes, Secrets]
        L2[Layer 2: Storage<br/>PostgreSQL, Redis, Vector DB]
        L3[Layer 3: Security<br/>Keycloak, Encryption, Audit]
        L4[Layer 4: Core Services<br/>Doc Ingester, Consensus, Blockchain]
        L5[Layer 5: Processing<br/>MinerU, YY-Chunker, Generators]
        L6[Layer 6: AI Services<br/>Agent Orchestrator, LLM Runtime]
        L7[Layer 7: Interface<br/>API Gateway, Web UI, Monitoring]
    end
    
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    L5 --> L6
    L6 --> L7
    
    class L1 fill:#e1f5fe
    class L2 fill:#f3e5f5
    class L3 fill:#e8f5e8
    class L4 fill:#fff3e0
    class L5 fill:#fce4ec
    class L6 fill:#f1f8e9
    class L7 fill:#e8eaf6
```

This comprehensive container interaction documentation provides the visual architecture diagrams needed to complete your DEAAP system design. Each subsystem diagram shows the specific container interactions, resource requirements, and network topology required for a production-ready deployment.

The diagrams integrate seamlessly with your existing microservices documentation, security specifications, and operational frameworks, providing the final piece needed for a complete enterprise AI agent platform architecture.
