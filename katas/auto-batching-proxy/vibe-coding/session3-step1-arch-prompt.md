As a senior software architect, read the requirements from "../rust-service-dev-interview.md" and review my top-level architecture. Decide if it's good for the job, find out if anything else should be defined. At the end, let's write together a task specification for a middle software developer.

# Architecture

## Main objects

- `Batcher`: the main logic
- `ActiveVector`: a wrapper over `Vec`, to limit its functionality and to notify listeners
- `Timer`: to notify when waiting is over

## Main logic

- The service handler accepts a query
- The handler passes the query to `Batcher`, which returns a `tokio::sync::oneshot` promise
- The handler awaits on the promise and returns the result

The result from `Batcher` includes http status code to report potential errors.

`Batcher` unpacks individual text fragments from the input and stores them in `ActiveVector`, together with the promise and the current timestamp.

Eventually, `Batcher` is notified with resolved embeddings. It should walk over the resolved batch, merge items of the same promise and resolve the promise.

## Background logic

`Batcher` waits for "time to embed a batch" events in an infinite loop. On an event:

- Check if the event valid: The timer is allowed to trigger earlier, or its notification can be for an old batch
- Get a new batch from `ActiveVec`. If it is empty, continue waiting for new events
- Otherwise, call the embedding service and pass the result to the main logic

## ActiveVec

`ActiveVec` wraps a Vec of records (text fragment, promise, timestamp, whatever) which are waiting to be batched.

Method `extend` adds several items of one promise to the vector.

Method `slice` extract a batch from the vector, up to the configured maximal size. Items of one promise should not be broken, consider "maximal size" as a recommendation in this case.

`ActiveVec` should trigger callbacks:

- To timer: There is a new first element. As a parameter, pass the injection timestamp
- To the background logic: The size of the vector is more or equal than the maximal size

## Timer

Timer is reset each time a new `set` is called.

## Special considerations

The service should eventually stop on SIGSTOP.

In the names for "Max Wait Time" and "Max Batch Size" include the word "soft", to adjust user expectations.

Return "503 Service Unavailable" if there are too much (let's say 10x of the configured vector max size) unprocessed items, including those which are in process by the upstream embedding service.


