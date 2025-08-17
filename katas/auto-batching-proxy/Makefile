.PHONY: build push clean test help

DOCKER_USERNAME = olpa
INFERENCE_IMAGE = $(DOCKER_USERNAME)/aubapr-inference
PROXY_IMAGE = $(DOCKER_USERNAME)/aubapr
VERSION = latest

help:
	@echo "Available targets:"
	@echo "  build          - Build both Docker images"
	@echo "  build-inference - Build inference service image"
	@echo "  build-proxy    - Build proxy service image"
	@echo "  push           - Push both images to DockerHub"
	@echo "  push-inference - Push inference image to DockerHub"
	@echo "  push-proxy     - Push proxy image to DockerHub"
	@echo "  test           - Run E2E tests"
	@echo "  clean          - Remove local Docker images"
	@echo "  up             - Start services with docker-compose"
	@echo "  down           - Stop services with docker-compose"
	@echo "  logs           - Show docker-compose logs"

build: build-inference build-proxy

build-inference:
	@echo "Building inference service image..."
	docker build -t $(INFERENCE_IMAGE):$(VERSION) ./aubapr-inference
	@echo "✅ Built $(INFERENCE_IMAGE):$(VERSION)"

build-proxy:
	@echo "Building proxy service image..."
	docker build -t $(PROXY_IMAGE):$(VERSION) ./aubapr
	@echo "✅ Built $(PROXY_IMAGE):$(VERSION)"

push: push-inference push-proxy

push-inference: build-inference
	@echo "Pushing inference service image to DockerHub..."
	docker push $(INFERENCE_IMAGE):$(VERSION)
	@echo "✅ Pushed $(INFERENCE_IMAGE):$(VERSION)"

push-proxy: build-proxy
	@echo "Pushing proxy service image to DockerHub..."
	docker push $(PROXY_IMAGE):$(VERSION)
	@echo "✅ Pushed $(PROXY_IMAGE):$(VERSION)"

clean:
	@echo "Removing local Docker images..."
	-docker rmi $(INFERENCE_IMAGE):$(VERSION)
	-docker rmi $(PROXY_IMAGE):$(VERSION)
	@echo "✅ Cleaned local images"

up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d
	@echo "✅ Services started"
	@echo "Proxy available at: http://localhost:8080"
	@echo "Inference service available at: http://localhost:8081"

down:
	@echo "Stopping services..."
	docker-compose down
	@echo "✅ Services stopped"

logs:
	docker-compose logs -f

test: up
	@echo "Waiting for services to start..."
	sleep 10
	@echo "Setting up test environment and running E2E tests..."
	cd tests && python3 -m venv venv || true
	cd tests && . venv/bin/activate && pip install -r requirements.txt > /dev/null 2>&1
	cd tests && . venv/bin/activate && python e2e_test.py --proxy-url http://localhost:8080 --inference-url http://localhost:8081

test-local:
	@echo "Running E2E tests (assuming services are already running)..."
	@echo "Setting up virtual environment if it doesn't exist..."
	cd tests && python3 -m venv venv || true
	cd tests && . venv/bin/activate && pip install -r requirements.txt > /dev/null 2>&1
	cd tests && . venv/bin/activate && python e2e_test.py --proxy-url http://localhost:8080 --inference-url http://localhost:8081

# Docker login helper
login:
	@echo "Please run: docker login"
	@echo "Then enter your DockerHub credentials"

# Quick development workflow
dev: build up test

# Production deployment workflow
deploy: build push
	@echo "✅ Images built and pushed to DockerHub"
	@echo "You can now deploy using:"
	@echo "  docker-compose up -d"