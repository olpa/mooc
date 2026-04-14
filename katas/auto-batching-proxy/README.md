# Auto-Batching Proxy E2E Testing Environment

This repository contains an end-to-end testing environment for an auto-batching proxy system, consisting of a Python inference service and a Rust proxy service. See [./rust-service-dev-interview.md](rust-service-dev-interview.md) for details.


# Fast Start

```
wget https://raw.githubusercontent.com/olpa/mooc/refs/heads/aubapr/katas/auto-batching-proxy/docker-compose.yaml
docker-compose up
```

On port 8081, there is a mock embedding service that treats calculations as a non-shareable limited resource: The requests are queued, and regardless of the batch size, one calculation takes 1 second.

On port 8080, there is the auto-batching proxy with the following configuration: collect batches of 32 requests, wait 100ms to collect a batch.

```
$ curl -H 'Content-type: application/json' -d '{"inputs": ["Hello, world!"]}' http://localhost:8081/embed
{"embeddings":[[0.8431372549019608,0.803921568627451,0.5529411764705883,0.803921568627451]]}

$ curl 127.0.0.1:8080/embed -X POST -d '{"inputs":["What is Vector Search?", "Hello, world!"]}' -H 'Content-Type: application/json'
{"embeddings":[[0.30196078431372547,-0.06666666666666667,0.8823529411764706,0.4196078431372549],[0.8431372549019608,0.803921568627451,0.5529411764705883,0.803921568627451]]}
```


# Performance Check

Run the script in the directory `tests`:

```
# Environment setup
# Already running: docker-compose up
cd tests
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt


# Run
python e2e_test.py
```

Here is the final summary:

```
  1 request via proxy:   1.11s
  1 request via inference:   1.01s

  3 requests via proxy:   1.11s
  3 requests via inference:   3.02s

 10 requests via proxy:   1.11s
 10 requests via inference:  10.03s

100 requests via proxy:   4.16s
100 requests via inference: 100.17s
```

The proxy timing of "1.1s" consists of:

- 100 ms wait to build a batch
- 1000 ms to get the answer from the inference server

For 100 requests the result is "4.1s":

- 3 seconds for 3 batches of 32 items
- 1.1 seconds for the incomplete batch of 8 items (100ms to wait, 1000ms the inference)


# Architecture of the Auto-Batching Proxy

## Main Objects

- `Batcher`: the main logic
- `Tray`: a room for waiting incoming requests
- `Timer`: to notify when waiting is over

## Main Logic

The service handler:

- Accept a query
- Pass the query to `Batcher`, which returns a `tokio::sync::oneshot` promise
- Await on the promise
- Eventually return the result

`Batcher`:

- Unpack individual text fragments from the input and store them in `Tray`, together with the promise
- Eventually get activated by a batch from `Tray`
- Call the upstream embedding service
- Walk through the batch and its calculated embeddings, merge items of the same promise and resolve the promise

## Background Logic

`Batcher` reacts to events in an infinite loop. Main events:

- Batch event from `Tray`: continue to the main logic
- Timer fires: Notify `Tray`, so that it sends a batch-event in return

## Tray

`Tray` wraps a Vec of records (one text fragment, promise, and whatever else we might need in the future) which are waiting to be batched.

As soon as the size reaches the configured soft max size, `Tray` sends the collection to `Batcher` and empties itself.

### Tray and Timer

When the first element is added, `Tray` sets the `Timer`. There is only one active timer at a time, which is the most recent one.

To handle race conditions between timer-based and size-based triggering, `Tray` holds the batch sequence number. A timer event with an outdated sequence number is ignored.


# Trace the Logic

The best starting points are the integration tests `test_multiple_batching_by_size` and `test_multiple_batching_by_wait` in `main.rs`:

```
$ cd aubapr
$ cargo test test_multiple_batching_by_size -- --nocapture
... debug output is here ....
$ cargo test test_multiple_batching_by_wait -- --nocapture
... debug output is here ....
```
