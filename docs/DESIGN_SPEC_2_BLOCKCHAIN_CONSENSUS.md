# System Design Specification 2: Blockchain Consensus & Validation

## Overview

This specification defines the blockchain-based consensus mechanism that validates data usage authorization across multiple Business Units (BUs) using Cartesi-inspired off-chain computation with on-chain verification.

## Component Architecture

```mermaid
graph TD
    subgraph "Consensus Coordination Layer"
        CM[Consensus Manager]
        BC[Blockchain Node]
        SM[Smart Contracts]
        CM --> BC
        BC --> SM
    end
    
    subgraph "BU Validator Ecosystem"
        VT[Validator Template]
        BU1[BU-1 Validator]
        BU2[BU-2 Validator]
        BUN[BU-N Validator]
        VT --> BU1
        VT --> BU2
        VT --> BUN
    end
    
    subgraph "Off-Chain Computation"
        CC1[Cartesi Container 1]
        CC2[Cartesi Container 2]
        CCN[Cartesi Container N]
        VL1[Validation Logic 1]
        VL2[Validation Logic 2]
        VLN[Validation Logic N]
        BU1 --> CC1
        BU2 --> CC2
        BUN --> CCN
        CC1 --> VL1
        CC2 --> VL2
        CCN --> VLN
    end
    
    subgraph "Consensus Resolution"
        AR[Aggregation Runtime]
        CR[Consensus Resolver]
        DR[Dispute Resolution]
        VL1 --> AR
        VL2 --> AR
        VLN --> AR
        AR --> CR
        CR --> DR
    end
    
    subgraph "Authorization Pipeline"
        AS[Authorization Service]
        TR[Traceability Registry]
        AC[Access Controller]
        CR --> AS
        AS --> TR
        AS --> AC
    end
    
    class CM,BC,SM fill:#e8f5e8
    class VT,BU1,BU2,BUN fill:#f3e5f5
    class CC1,CC2,CCN,VL1,VL2,VLN fill:#e1f5fe
    class AR,CR,DR fill:#fff3e0
    class AS,TR,AC fill:#ffebee
```

## Container Specifications

### 1. Blockchain Node Service (`blockchain`)

**Base Image**: `ethereum/client-go:v1.13.0`

**Ports**:
- `8545:8545` (JSON-RPC)
- `30303:30303` (P2P)
- `8546:8546` (WebSocket)

**Volumes**:
- `./blockchain/data:/data`
- `./blockchain/genesis.json:/genesis.json`
- `./blockchain/keystore:/keystore`

**Environment Variables**:
```env
NETWORK_ID=31337
CHAIN_ID=31337
BLOCK_TIME=12
GAS_LIMIT=30000000
CONSENSUS_ALGORITHM=proof-of-authority
VALIDATOR_ACCOUNTS=3
```

### 2. Consensus Manager Service (`consensus-manager`)

**Base Image**: `node:18-alpine`

**Ports**:
- `8760:8760` (Management API)
- `8761:8761` (WebSocket Events)

**Volumes**:
- `./services/consensus-manager:/app`
- `./blockchain/contracts:/app/contracts`
- `./data/consensus:/app/data`

**Environment Variables**:
```env
BLOCKCHAIN_RPC=http://blockchain:8545
SMART_CONTRACT_ADDRESS=${CONSENSUS_CONTRACT_ADDRESS}
VALIDATOR_REGISTRY_ADDRESS=${VALIDATOR_REGISTRY_ADDRESS}
MIN_VALIDATORS=3
CONSENSUS_THRESHOLD=0.67
VALIDATION_TIMEOUT=300
DISPUTE_WINDOW=3600
```

### 3. BU Validator Template (`bu-validator-template`)

**Base Image**: `cartesi/playground:0.5.0`

**Ports**:
- `8762:8762` (Validator API)
- `5005:5005` (Cartesi Inspect)

**Volumes**:
- `./services/bu-validator:/app`
- `./validator-logic/${BU_NAME}:/app/logic`
- `./data/validator/${BU_NAME}:/app/data`

**Environment Variables**:
```env
BU_NAME=${BU_NAME}
BU_WALLET_ADDRESS=${BU_WALLET_ADDRESS}
BU_PRIVATE_KEY_PATH=/run/secrets/bu_private_key
VALIDATION_LOGIC_PATH=/app/logic/validation.py
CARTESI_MODE=production
BLOCKCHAIN_RPC=http://blockchain:8545
CONSENSUS_MANAGER_URL=http://consensus-manager:8760
VALIDATOR_STAKE_AMOUNT=1000000000000000000
```

### 4. Cartesi Runtime Service (`cartesi-runtime`)

**Base Image**: `cartesi/rollups-node:1.5.1`

**Ports**:
- `5006:5006` (Cartesi Node)
- `8080:8080` (GraphQL)

**Volumes**:
- `./services/cartesi-runtime:/opt/cartesi/dapp`
- `./validator-logic:/opt/cartesi/logic`
- `cartesi_data:/opt/cartesi/share`

**Environment Variables**:
```env
CARTESI_BLOCKCHAIN_RPC_URL=http://blockchain:8545
CARTESI_EPOCH_DURATION=3600
CARTESI_BLOCKCHAIN_BLOCK_TIMEOUT=60
CARTESI_LOG_LEVEL=info
CARTESI_CONTRACTS_APPLICATION_ADDRESS=${CARTESI_APPLICATION_ADDRESS}
```

### 5. Authorization Service (`authorization`)

**Base Image**: `python:3.10-slim`

**Ports**:
- `8763:8763` (Authorization API)

**Volumes**:
- `./services/authorization:/app`
- `./data/auth:/app/data`
- `./configs/policies:/app/policies`

**Environment Variables**:
```env
DATABASE_URL=postgresql://user:pass@postgres:5432/authorization
REDIS_URL=redis://redis:6379
BLOCKCHAIN_RPC=http://blockchain:8545
CONSENSUS_CONTRACT=${CONSENSUS_CONTRACT_ADDRESS}
POLICY_ENGINE=opa
AUDIT_ENABLED=true
```

## Smart Contract Architecture

### 1. Consensus Contract (`ConsensusManager.sol`)

```solidity
contract ConsensusManager {
    struct ValidationRequest {
        bytes32 dataHash;
        address[] validators;
        mapping(address => bytes32) votes;
        uint256 startTime;
        bool resolved;
        bool approved;
    }
    
    struct ValidatorInfo {
        address validator;
        string buName;
        uint256 stake;
        bool active;
    }
    
    mapping(bytes32 => ValidationRequest) public validationRequests;
    mapping(address => ValidatorInfo) public validators;
    
    uint256 public minValidators = 3;
    uint256 public consensusThreshold = 67; // 67%
    uint256 public validationTimeout = 3600; // 1 hour
    
    event ValidationRequested(bytes32 indexed requestId, bytes32 indexed dataHash);
    event VoteSubmitted(bytes32 indexed requestId, address indexed validator, bytes32 vote);
    event ConsensusReached(bytes32 indexed requestId, bool approved);
    event ValidatorRegistered(address indexed validator, string buName);
    
    function requestValidation(bytes32 dataHash, address[] memory selectedValidators) external;
    function submitVote(bytes32 requestId, bytes32 voteHash, bytes calldata proof) external;
    function resolveConsensus(bytes32 requestId) external;
    function registerValidator(string memory buName, uint256 stake) external;
}
```

### 2. Data Authorization Contract (`DataAuthorization.sol`)

```solidity
contract DataAuthorization {
    struct DataAsset {
        bytes32 assetHash;
        address owner;
        string assetType; // "lora", "embedding", "dataset"
        string metadata;
        bool authorized;
        uint256 authorizationTime;
        bytes32 consensusRequestId;
    }
    
    mapping(bytes32 => DataAsset) public dataAssets;
    mapping(address => mapping(bytes32 => bool)) public userPermissions;
    
    event DataAssetRegistered(bytes32 indexed assetHash, address indexed owner, string assetType);
    event AuthorizationGranted(bytes32 indexed assetHash, bytes32 indexed consensusRequestId);
    event PermissionGranted(bytes32 indexed assetHash, address indexed user);
    
    function registerDataAsset(bytes32 assetHash, string memory assetType, string memory metadata) external;
    function authorizeData(bytes32 assetHash, bytes32 consensusRequestId) external;
    function grantPermission(bytes32 assetHash, address user) external;
}
```

## Validation Logic Framework

### Base Validation Interface

```python
class BaseValidator:
    def __init__(self, bu_config: dict):
        self.bu_name = bu_config['name']
        self.policies = bu_config['policies']
        self.weights = bu_config['weights']
    
    def validate_data_usage(self, request: ValidationRequest) -> ValidationResult:
        """
        Core validation method - must be implemented by each BU
        """
        raise NotImplementedError
    
    def check_data_sensitivity(self, data_metadata: dict) -> bool:
        """Check if data contains sensitive information"""
        pass
    
    def verify_usage_purpose(self, purpose: str, requester: str) -> bool:
        """Verify the legitimacy of data usage purpose"""
        pass
    
    def calculate_risk_score(self, request: ValidationRequest) -> float:
        """Calculate risk score for the validation request"""
        pass
```

### BU-Specific Validation Examples

#### Example 1: Financial BU Validator

```python
class FinancialBUValidator(BaseValidator):
    def validate_data_usage(self, request: ValidationRequest) -> ValidationResult:
        # Check compliance with financial regulations
        compliance_score = self.check_financial_compliance(request.data_metadata)
        
        # Verify data classification
        classification = self.classify_financial_data(request.data_content)
        
        # Check purpose legitimacy
        purpose_valid = self.verify_financial_purpose(request.purpose)
        
        # Calculate overall approval
        approval = (
            compliance_score > 0.8 and 
            classification['risk_level'] <= 'medium' and 
            purpose_valid
        )
        
        return ValidationResult(
            approved=approval,
            confidence=compliance_score,
            reasons=self.generate_reasoning(compliance_score, classification, purpose_valid),
            metadata={'bu': 'financial', 'compliance_score': compliance_score}
        )
```

#### Example 2: Legal BU Validator

```python
class LegalBUValidator(BaseValidator):
    def validate_data_usage(self, request: ValidationRequest) -> ValidationResult:
        # Check for legal privilege
        privilege_check = self.check_attorney_client_privilege(request.data_metadata)
        
        # Verify GDPR compliance
        gdpr_compliance = self.verify_gdpr_compliance(request.data_content)
        
        # Check data retention policies
        retention_valid = self.check_retention_policy(request.usage_duration)
        
        approval = (
            not privilege_check['privileged'] and 
            gdpr_compliance['compliant'] and 
            retention_valid
        )
        
        return ValidationResult(
            approved=approval,
            confidence=0.95 if approval else 0.1,
            reasons=self.generate_legal_reasoning(privilege_check, gdpr_compliance, retention_valid),
            metadata={'bu': 'legal', 'gdpr_compliance': gdpr_compliance}
        )
```

## Consensus Algorithm

### Validation Process Flow

1. **Request Initiation**
   ```
   Agent Request → Authorization Service → Consensus Manager → Smart Contract
   ```

2. **Validator Selection**
   ```
   Smart Contract → Select Validators → Notify BU Validators
   ```

3. **Off-Chain Validation**
   ```
   BU Validators → Execute Custom Logic → Generate Proof → Submit Vote
   ```

4. **Consensus Resolution**
   ```
   Collect Votes → Calculate Consensus → Update Smart Contract → Authorize Data
   ```

### Consensus Mechanisms

#### Weighted Voting
- Each BU has a stake-weighted vote
- Minimum participation threshold (e.g., 3 validators)
- Consensus threshold (e.g., 67% weighted approval)

#### Dispute Resolution
- Challenge period for disputed decisions
- Evidence submission and review
- Economic penalties for malicious behavior

## Data Flow Specification

### Validation Request Flow

```mermaid
sequenceDiagram
    participant User
    participant AS as Authorization Service
    participant CM as Consensus Manager
    participant SC as Smart Contract
    participant BV as BU Validators
    participant CR as Cartesi Runtime

    User->>AS: Request data usage authorization
    AS->>CM: Initiate validation request
    CM->>SC: Record validation request
    SC->>BV: Notify selected validators
    BV->>CR: Execute validation logic
    CR->>BV: Return validation results
    BV->>SC: Submit votes with proofs
    SC->>CM: Check consensus threshold
    CM->>AS: Return authorization decision
    AS->>User: Grant/deny access
```

## Performance & Scalability

### Throughput Requirements
- **Validation Requests**: 100 requests/hour
- **Consensus Resolution**: 15 minutes average
- **Validator Response**: 5 minutes maximum

### Scalability Measures
- **Horizontal Scaling**: Add more BU validators
- **Sharding**: Partition by data domain
- **Caching**: Cache frequent validation results

## Security & Trust Model

### Threat Model
- **Malicious Validators**: Economic penalties and reputation system
- **Data Tampering**: Cryptographic proofs and merkle trees
- **Collusion**: Random validator selection and monitoring

### Trust Assumptions
- **Majority Honest**: At least 67% of validators are honest
- **Economic Rationality**: Validators act in their economic interest
- **Cryptographic Security**: Hash functions and digital signatures are secure
