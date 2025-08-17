# Auto-Batching Proxy E2E Testing Environment

This repository contains an end-to-end testing environment for an auto-batching proxy system, consisting of a Python inference service and a Rust proxy service.

## Architecture

- **aubapr-inference**: Python FastAPI service that provides text embeddings using CRC32-based calculations
- **aubapr**: Rust proxy service that forwards requests to the inference service (batching logic to be implemented)
- **E2E Tests**: Performance comparison tests to validate batching effectiveness

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Make (optional, for convenient commands)
- Python 3.11+ (for running tests locally)

### Build and Run Services

```bash
# Build Docker images
make build

# Start services
make up

# Run E2E tests
make test

# Stop services
make down
```

### Manual Docker Compose

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down
```

## Service Endpoints

### Proxy Service (Port 8080)
- **Embed Endpoint**: `POST http://localhost:8080/embed`
- **Health Check**: `GET http://localhost:8080/health`

### Inference Service (Port 8081)
- **Embed Endpoint**: `POST http://localhost:8081/embed`
- **Health Check**: `GET http://localhost:8081/health`

## API Usage Examples

### Basic Embedding Request

```bash
# Via Proxy Service
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello, world!", "How are you?"]}'

# Direct to Inference Service
curl -X POST http://localhost:8081/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello, world!", "How are you?"]}'
```

### Expected Response Format

```json
{
  "embeddings": [
    [-0.1234, 0.5678, -0.9012, 0.3456],
    [0.7890, -0.2345, 0.6789, -0.0123]
  ]
}
```

## Performance Testing

### Running E2E Tests

**Prerequisites**: Services must be running first:
```bash
# Start services if not already running
make up
# OR
docker-compose up -d
```

**Option 1: Using Make (auto-handles virtual environment)**
```bash
# With services auto-start (starts services + runs tests)
make test

# With services already running (recommended)
make test-local
```

> **Note**: The make commands automatically create and manage a virtual environment in `tests/venv/` and install dependencies. No manual setup required!

**Option 2: Manual test execution with virtual environment**
```bash
cd tests

# Create and activate virtual environment (first time only)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python e2e_test.py --proxy-url http://localhost:8080 --inference-url http://localhost:8081

# Deactivate virtual environment when done
deactivate
```

**Option 3: Using system Python (not recommended)**
```bash
cd tests
pip install --break-system-packages -r requirements.txt
python e2e_test.py --proxy-url http://localhost:8080 --inference-url http://localhost:8081
```

### Test Scenarios

The E2E tests compare performance across different scenarios:

1. **Single Request via Proxy**: Baseline proxy performance
2. **10 Concurrent Requests via Proxy**: Current concurrent handling
3. **10 Sequential Requests via Proxy**: Sequential processing baseline
4. **Direct Inference Service Tests**: Baseline without proxy overhead

### Expected Results

**Current Behavior (Stub Implementation):**
- Single request: ~1 second (1s delay per request)
- 10 concurrent requests: ~10 seconds (processed sequentially, each takes 1s)
- 10 sequential requests: ~10 seconds (processed one after another)

**Future Behavior (With Batching):**
- Single request: ~1 second
- 10 concurrent requests: ~1-2 seconds (batched together)
- Significant performance improvement for concurrent workloads

## Development

### Building Individual Services

```bash
# Build inference service
make build-inference

# Build proxy service
make build-proxy
```

### Publishing to DockerHub

```bash
# Login to DockerHub
docker login

# Build and push both images
make deploy
```

### Development Workflow

```bash
# Quick development cycle
make dev  # Equivalent to: make build up test
```

### Viewing Logs

```bash
# All services
make logs

# Individual service
docker-compose logs -f aubapr
docker-compose logs -f aubapr-inference
```

## Service Details

### Inference Service (aubapr-inference)

**Technology**: Python FastAPI  
**Port**: 8080 (internal), 8081 (external)

**Features**:
- CRC32-based 4-dimensional embeddings
- 1-second delay per request (regardless of input count)
- Sequential request processing (no concurrent handling)
- Compatible with text-embeddings-inference API

**Configuration**:
- `PYTHONUNBUFFERED=1`: Immediate log output

### Proxy Service (aubapr)

**Technology**: Rust with Axum framework  
**Port**: 8080

**Features**:
- Request forwarding to inference service
- Error handling and logging
- Health check endpoint
- Future: Auto-batching logic

**Configuration**:
- `INFERENCE_URL`: Target inference service URL
- `RUST_LOG`: Logging level

## Testing Strategy

### Manual Testing

```bash
# Test proxy health
curl http://localhost:8080/health

# Test inference health
curl http://localhost:8081/health

# Single embedding via proxy
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Test message"]}'

# Single embedding direct to inference
curl -X POST http://localhost:8081/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Test message"]}'
```

### Performance Benchmarking

The included E2E tests provide comprehensive performance analysis:

- **Baseline Measurements**: Direct inference service performance
- **Proxy Overhead**: Additional latency introduced by proxy
- **Concurrency Handling**: How well the system handles multiple simultaneous requests
- **Future Validation**: Framework for testing batching improvements

## Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs
```

**Connection refused:**
```bash
# Ensure services are healthy
curl http://localhost:8080/health
curl http://localhost:8081/health

# Check docker network
docker network ls
```

**Tests failing:**
```bash
# Verify services are running
make up

# Wait for services to be ready
sleep 10

# Run tests manually with virtual environment
cd tests
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python e2e_test.py --proxy-url http://localhost:8080 --inference-url http://localhost:8081
```

**Python environment issues:**
```bash
# If you get "externally-managed-environment" error:
# Option 1: Use virtual environment (recommended)
cd tests
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option 2: Use system packages (if available)
sudo apt install python3-aiohttp

# Option 3: Force install (not recommended)
pip install --break-system-packages -r requirements.txt
```

**Make command not working:**
```bash
# If make test-local fails, run manually:
cd tests
source venv/bin/activate  # If venv exists
python e2e_test.py --proxy-url http://localhost:8080 --inference-url http://localhost:8081
```

### Port Conflicts

If ports 8080 or 8081 are in use:

1. Stop conflicting services
2. Or modify ports in `docker-compose.yaml`
3. Update test URLs accordingly

## Future Enhancements

This implementation provides the foundation for developing actual batching logic:

1. **Batching Implementation**: Accumulate requests and batch them
2. **Configuration**: Max wait time and batch size parameters
3. **Metrics**: Request timing and batching efficiency monitoring
4. **Load Testing**: Scale testing with higher concurrent loads

## Project Structure

```
.
├── aubapr-inference/          # Python inference service
│   ├── app.py                 # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile            # Container definition
├── aubapr/                   # Rust proxy service
│   ├── src/main.rs           # Rust application
│   ├── Cargo.toml            # Rust dependencies
│   └── Dockerfile            # Container definition
├── tests/                    # E2E tests
│   ├── e2e_test.py           # Test implementation
│   └── requirements.txt      # Test dependencies
├── Makefile                  # Build automation
├── docker-compose.yaml       # Service orchestration
├── README.md                 # This file
└── plan-e2e.md              # Implementation plan
```