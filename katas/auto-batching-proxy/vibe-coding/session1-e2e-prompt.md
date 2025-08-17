As a senior architect, create a workplan to implement an e2e testing environment for an auto-batching proxy (see "rust-service-dev-interview.md" for details).

Use my high-level plan as an inspiration:

- Create a Python inference service
  - provide a batch interface, compatible "https://github.com/huggingface/text-embeddings-inference"
  - do not care about the model parameter and anything alse except input texts
  - to calculate an embedding vector:
    - calculate crc32 from the input string, use 4 bytes as 4 vector items
  - Implement limits:
    - add 1 second delay to one vector calculation
    - vector calculations should be done one by one, without parallel work

- Create a stub for the future auto-batching-proxy. It should do nothing yet, except proxing requests to the Python inference service.

- Create a e2e test
  - make 10 requests to the auto-batching-proxy, remember the elapsed time
  - make 1 request to the auto-batching-proxy, remember the elapsed time
  - compare the times. They should be of the same magnitude, not 10 times difference

The Python service should be called "aubapr-inference", the Rust service should be called "aubapr". Create a Makefile do build Docker images and put them to dockerhub under my name "olpa". Finally, create a docker-compose.yaml to run them and a README.md how to run the services and make a test using curl, both against the proxy and the inference service.

Put the implementation plan and the instructions for the senior software developer to the file "plan-e2e.md". The steps should be independant, you should provide enough information to re-start working at a step.

Ask me for details if you have questions.
