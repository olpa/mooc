# Auto-Batching Proxy E2E Testing Environment

This repository contains an end-to-end testing environment for an auto-batching proxy system, consisting of a Python inference service and a Rust proxy service.

# TODO: Fast Start


# Architecture of the Auto-Batching Proxy

## Main objects

- `Batcher`: the main logic
- `Tray`: a room for waiting incoming requests
- `Timer`: to notify when waiting is over

## Main logic

The service handler:

- Accept a query
- Pass the query to `Batcher`, which returns a `tokio::sync::oneshot` promise
- Await on the promise
- Eventually return the result

`Batcher`:

- Unpack individual text fragments from the input and store them in `Tray`, together with the promise
- Eventually be activated by a batch from `Tray`
- Call the upstream embedding service
- Walk over the batch and its calculated embeddings, merge items of the same promise and resolve the promise

## Background logic

`Batcher` reacts to events in an infinite loop. Main events:

- Batch event from `Tray`: continue to the main logic
- Timer is firing: Notify `Tray`, so that in its turn it sends a batch-event

## Tray

`Tray` wraps a Vec of records (one text fragment, promise, whatever we'll need in future) which are waiting to be batched.

As soon as the size reaches the configured soft max size, `Tray` sends the collection to `Batcher` and empties itself.

### Tray and Timer

When a first element is added, `Tray` sets the `Timer`. There is only one active timer at once, the last one.

To handle race conditions of timer-based and size-based triggering, `Tray` holds the batch sequence number. A timer event with an old sequence number is ignored.


# TODO

Move to a better place

## Repository structure

- **aubapr-inference**: Python FastAPI service that provides text embeddings using CRC32-based calculations
- **aubapr**: Rust proxy service that forwards requests to the inference service (batching logic to be implemented)
- **E2E Tests**: Performance comparison tests to validate batching effectiveness

## Quick Start

### Prerequisites

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