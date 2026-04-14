# E2E Testing Environment Implementation Plan

## Overview
This plan outlines the implementation of an end-to-end testing environment for an auto-batching proxy system. The system consists of a Python inference service, a Rust proxy service, and comprehensive testing infrastructure.

## Architecture
- **aubapr-inference**: Python service providing embeddings calculation with artificial delays
- **aubapr**: Rust proxy service (initial stub implementation)
- **E2E Tests**: Performance comparison tests
- **Infrastructure**: Docker, Makefile, docker-compose

## Implementation Steps

### Step 1: Python Inference Service (aubapr-inference)
**Objective**: Create a mock inference service compatible with text-embeddings-inference API

**Requirements**:
- HTTP REST API with `/embed` endpoint
- Accept POST requests with JSON payload: `{"inputs": ["text1", "text2", ...]}`
- Return embeddings in compatible format
- Use CRC32 hash of input text to generate 4-dimensional vectors
- Add 1-second delay per request (regardless of number of inputs)
- Process requests sequentially (no concurrent handling)

**Implementation Details**:
- Use FastAPI framework for HTTP server
- Endpoint: `POST /embed`
- Input format: `{"inputs": List[str]}`
- Output format: `{"embeddings": List[List[float]]}`
- Vector calculation: CRC32(text) → 4 bytes → 4 float values
- Sequential request processing with 1s delay per request

**Files to create**:
- `aubapr-inference/app.py` - Main FastAPI application
- `aubapr-inference/requirements.txt` - Python dependencies
- `aubapr-inference/Dockerfile` - Container definition

**Dependencies**:
- FastAPI
- uvicorn
- zlib (for CRC32)

### Step 2: Rust Auto-Batching Proxy Stub (aubapr)
**Objective**: Create a basic proxy that forwards requests to the inference service

**Requirements**:
- HTTP server accepting same API as inference service
- Forward all requests to aubapr-inference service
- No batching logic yet (just proxying)
- Compatible with text-embeddings-inference API

**Implementation Details**:
- Use axum framework for HTTP server
- Proxy endpoint: `POST /embed`
- Forward requests to `http://aubapr-inference:8080/embed`
- Preserve request/response format
- Basic error handling

**Files to create**:
- `aubapr/src/main.rs` - Main application
- `aubapr/Cargo.toml` - Rust dependencies
- `aubapr/Dockerfile` - Container definition

**Dependencies**:
- axum
- tokio
- reqwest
- serde/serde_json

### Step 3: E2E Test Implementation
**Objective**: Create performance comparison tests

**Requirements**:
- Test 10 concurrent requests vs 1 single request
- Measure and compare execution times
- Verify that proxy batching provides performance benefits
- Test both proxy and inference service directly

**Implementation Details**:
- Python test script using asyncio/aiohttp
- Test scenarios:
  1. 10 concurrent requests to proxy
  2. 1 single request to proxy
  3. Direct calls to inference service for baseline
- Time measurement and comparison
- Assert that batched requests don't take 10x longer

**Files to create**:
- `tests/e2e_test.py` - Main test implementation
- `tests/requirements.txt` - Test dependencies

### Step 4: Docker Infrastructure
**Objective**: Create containerized deployment setup

**Makefile Requirements**:
- Build Docker images for both services
- Tag images with DockerHub username "olpa"
- Push images to DockerHub
- Clean and rebuild options

**docker-compose.yaml Requirements**:
- Define both services with proper networking
- Expose appropriate ports
- Set up service dependencies
- Volume mounts if needed

**Files to create**:
- `Makefile` - Build and deployment automation
- `docker-compose.yaml` - Service orchestration

### Step 5: Documentation
**Objective**: Create comprehensive usage instructions

**README.md Requirements**:
- Service overview and architecture
- Build and deployment instructions
- Testing procedures
- curl examples for both services
- Performance testing instructions

**Files to create**:
- `README.md` - Main documentation

## Testing Strategy

### Unit Tests
- Test vector calculation logic in Python service
- Test proxy forwarding in Rust service

### Integration Tests
- Test API compatibility between services
- Test error handling and edge cases

### E2E Performance Tests
- Measure baseline performance (direct inference calls)
- Measure proxy performance with single requests
- Measure proxy performance with concurrent requests
- Compare and validate performance improvements

## Expected Outcomes

### Performance Expectations
- Direct inference: ~1 second per request (sequential)
- 10 individual requests via proxy: ~10 seconds total
- Future batched requests via proxy: ~1-2 seconds total (after batching implementation)

### Success Criteria
- All services build and run successfully
- API compatibility maintained
- Performance tests demonstrate expected behavior
- Documentation allows independent setup and testing

## File Structure
```
.
├── aubapr-inference/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── aubapr/
│   ├── src/main.rs
│   ├── Cargo.toml
│   └── Dockerfile
├── tests/
│   ├── e2e_test.py
│   └── requirements.txt
├── Makefile
├── docker-compose.yaml
├── README.md
└── plan-e2e.md
```

## Implementation Notes

### Dependencies and Considerations
- Ensure network connectivity between containers
- Handle service startup timing in docker-compose
- Implement proper logging for debugging
- Add health check endpoints for monitoring

### Future Enhancements (Post-MVP)
- Implement actual batching logic in Rust proxy
- Add configuration for max wait time and batch size
- Implement more sophisticated timing controls
- Add metrics and monitoring

This plan provides sufficient detail for independent implementation of each step while maintaining clear separation of concerns and testable milestones.