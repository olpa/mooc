# Implementation Specification: Auto-Batching Proxy Service

## Overview

Transform the existing stub proxy service into an auto-batching proxy that aggregates individual text embedding requests into efficient batches before forwarding to a HuggingFace text-embeddings-inference service.

## Architecture

### Main Objects

- `Batcher`: the main logic
- `ActiveVector`: a wrapper over `Vec`, to limit its functionality and to notify listeners
- `Timer`: to notify when waiting is over

### Main Logic

- The service handler accepts a query
- The handler passes the query to `Batcher`, which returns a `tokio::sync::oneshot` promise
- The handler awaits on the promise and returns the result

The result from `Batcher` includes http status code to report potential errors.

`Batcher` unpacks individual text fragments from the input and stores them in `ActiveVector`, together with the promise and the current timestamp.

Eventually, `Batcher` is notified with resolved embeddings. It should walk over the resolved batch, merge items of the same promise and resolve the promise.

### Background Logic

`Batcher` waits for "time to embed a batch" events in an infinite loop. On an event:

- Check if the event valid: The timer is allowed to trigger earlier, or its notification can be for an old batch
- Get a new batch from `ActiveVec`. If it is empty, continue waiting for new events
- Otherwise, call the embedding service and pass the result to the main logic

### ActiveVec

`ActiveVec` wraps a Vec of records (text fragment, promise, timestamp, whatever) which are waiting to be batched.

Method `extend` adds several items of one promise to the vector.

Method `slice` extract a batch from the vector, up to the configured maximal size. Items of one promise should not be broken, consider "maximal size" as a recommendation in this case.

`ActiveVec` should trigger callbacks:

- To timer: There is a new first element. As a parameter, pass the injection timestamp
- To the background logic: The size of the vector is more or equal than the maximal size

### Timer

Timer is reset each time a new `set` is called.

### Special Considerations

The service should eventually stop on SIGSTOP.

In the names for "Max Wait Time" and "Max Batch Size" include the word "soft", to adjust user expectations.

Return "503 Service Unavailable" if there are too much (let's say 10x of the configured vector max size) unprocessed items, including those which are in process by the upstream embedding service.

## Current Implementation Status

The existing codebase provides:
- ✅ HTTP server using `axum` framework
- ✅ Basic `/embed` and `/health` endpoints 
- ✅ Request/response structures (`EmbedRequest`, `EmbedResponse`)
- ✅ HTTP client setup with `reqwest`
- ✅ Error handling and upstream error propagation
- ✅ Comprehensive test suite with `wiremock`
- ✅ Proper Rust project structure with linting

## Implementation Tasks

### 1. Configuration Management (`src/config.rs`)

Create a new configuration module:

```rust
pub struct Config {
    pub soft_max_wait_time_ms: u64,
    pub soft_max_batch_size: usize,
    pub inference_url: String,
    pub max_queue_size: usize, // 10x soft_max_batch_size
}
```

- Load from environment variables with sensible defaults
- `SOFT_MAX_WAIT_TIME_MS` (default: 100ms)
- `SOFT_MAX_BATCH_SIZE` (default: 32) 
- `INFERENCE_URL` (already exists)

### 2. Batch Processing Types (`src/types.rs`)

Define core data structures:

```rust
pub struct BatchItem {
    pub text: String,
    pub request_id: Uuid,
    pub sender: oneshot::Sender<BatchResult>,
    pub timestamp: Instant,
}

pub type BatchResult = Result<Vec<f64>, BatchError>;

#[derive(Debug)]
pub enum BatchError {
    UpstreamError(StatusCode, String),
    Timeout,
    ServiceUnavailable,
}
```

### 3. ActiveVector Implementation (`src/active_vector.rs`)

```rust
pub struct ActiveVector {
    items: Vec<BatchItem>,
    timer_callback: Option<tokio::sync::mpsc::UnboundedSender<Instant>>,
    batch_callback: Option<tokio::sync::mpsc::UnboundedSender<()>>,
}

impl ActiveVector {
    pub fn extend(&mut self, items: Vec<BatchItem>) { ... }
    pub fn slice(&mut self, max_size: usize) -> Vec<BatchItem> { ... }
    pub fn len(&self) -> usize { ... }
    pub fn set_callbacks(&mut self, timer_tx: ..., batch_tx: ...) { ... }
}
```

### 4. Timer Implementation (`src/timer.rs`)

```rust
pub struct Timer {
    current_task: Option<tokio::task::JoinHandle<()>>,
    sender: tokio::sync::mpsc::UnboundedSender<()>,
}

impl Timer {
    pub fn set(&mut self, duration: Duration, timestamp: Instant) { ... }
    pub fn receiver(&self) -> tokio::sync::mpsc::UnboundedReceiver<()> { ... }
}
```

### 5. Batcher Implementation (`src/batcher.rs`)

```rust
pub struct Batcher {
    active_vector: Mutex<ActiveVector>,
    timer: Mutex<Timer>,
    client: reqwest::Client,
    config: Config,
}

impl Batcher {
    pub async fn submit_request(&self, inputs: Vec<String>) -> Result<Vec<Vec<f64>>, BatchError> { ... }
    pub async fn run_background_processor(&self) { ... }
}
```

### 6. Update Main Service (`src/main.rs`)

Modify existing code to:
- Replace direct HTTP forwarding with `Batcher::submit_request()` 
- Initialize `Batcher` and start background processor
- Add graceful shutdown handling (SIGTERM)
- Return 503 when queue exceeds `max_queue_size`

### 7. Error Handling Updates

Update the existing error handling to work with batching:
- Map `BatchError` to appropriate HTTP status codes
- Preserve upstream error details 
- Handle partial batch failures

### 8. Testing Updates

Extend existing test suite:
- Test batching behavior with multiple concurrent requests
- Test timeout scenarios  
- Test backpressure (503 responses)
- Test graceful shutdown
- Keep existing upstream error tests

## Key Implementation Notes

- **Preserve existing API**: The `/embed` endpoint signature remains unchanged
- **Maintain error compatibility**: Upstream errors should propagate with same status codes
- **Thread safety**: Use `Arc<Mutex<>>` or `tokio::sync::Mutex` for shared state
- **Request correlation**: Ensure each input string returns to the correct original request
- **Graceful degradation**: Handle upstream service failures without crashing
- **No breaking changes**: All existing tests should continue to pass

## Success Criteria

1. ✅ Service batches requests efficiently within time/size constraints
2. ✅ Individual requests receive correct embeddings in proper order  
3. ✅ 503 responses when overwhelmed (>10x batch size queued)
4. ✅ Graceful shutdown preserves in-flight requests
5. ✅ All existing tests continue to pass
6. ✅ New tests validate batching behavior
7. ✅ Performance improvement demonstrated in benchmarks

## Existing Code Preservation

- Keep all current dependencies in `Cargo.toml`
- Maintain existing strict linting rules
- Preserve current project structure and naming
- Keep existing test infrastructure (`axum-test`, `wiremock`)
- Maintain current error response formats