.PHONY: help build up down down-dev restart shell logs clean test lint
.PHONY: build-gpu push-gpu build-sif up-gpu-docker-hub up-gpu-sif-file
.PHONY: build-hierarchical build-gpu-hierarchical test-mineland test-mineland-integration up-hierarchical

# Load .env file if it exists
-include .env
export

# Enable Docker BuildKit for faster builds and cache mounts
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Docker registry configuration (set in .env or environment)
DOCKER_REGISTRY ?= docker.io
DOCKER_USERNAME ?= $(shell whoami)
DOCKER_IMAGE_NAME ?= empo
GPU_IMAGE_TAG ?= gpu-latest
SIF_FILE ?= empo-gpu.sif

# Default target
help:
	@echo "EMPO Development Commands"
	@echo "========================="
	@echo "Local Development:"
	@echo "  make build          - Build Docker image (CPU)"
	@echo "  make build-hierarchical - Build Docker image with hierarchical deps (Ollama client, MineLand)"
	@echo "  make up             - Start development environment (auto-detects GPU)"
	@echo "  make up-hierarchical - Start with Ollama server container (for LLM inference)"
	@echo "  make down           - Stop development environment"
	@echo "  make restart        - Restart development environment"
	@echo "  make shell          - Open shell in container"
	@echo "  make logs           - Show container logs"
	@echo "  make train          - Run training script"
	@echo "  make example        - Run simple example"
	@echo "  make test           - Run tests"
	@echo "  make test-mineland  - Test MineLand installation (basic import tests)"
	@echo "  make test-mineland-integration - Test MineLand + Ollama vision (full integration)"
	@echo "  make lint           - Run linters"
	@echo "  make clean          - Clean up outputs and cache"
	@echo ""
	@echo "Cluster Deployment (GPU):"
	@echo "  make build-gpu              - Build GPU Docker image only"
	@echo "  make build-gpu-hierarchical - Build GPU image with hierarchical deps"
	@echo "  make up-gpu-docker-hub      - Build GPU image and push to Docker Hub"
	@echo "  make up-gpu-sif-file        - Build GPU image and convert to SIF file locally"
	@echo "  make push-gpu               - Push GPU image to Docker Hub"
	@echo "  make build-sif              - Convert GPU Docker image to SIF file"
	@echo ""
	@echo "Configuration via .env or environment:"
	@echo "  DOCKER_USERNAME      - Docker Hub username (default: $(DOCKER_USERNAME))"
	@echo "  DOCKER_REGISTRY      - Docker registry (default: $(DOCKER_REGISTRY))"
	@echo "  GPU_IMAGE_TAG        - GPU image tag (default: $(GPU_IMAGE_TAG))"
	@echo "  SIF_FILE             - Output SIF filename (default: $(SIF_FILE))"
	@echo "  HIERARCHICAL_MODE    - Enable hierarchical deps in build (default: false)"

# Docker Compose commands
build:
	@docker compose build

up:
	@echo "Starting development environment..."
	@# Set USER_ID and GROUP_ID if not already set
	@if [ -z "$$USER_ID" ]; then export USER_ID=$$(id -u); fi; \
	if [ -z "$$GROUP_ID" ]; then export GROUP_ID=$$(id -g); fi; \
	if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
		echo "✓ GPU detected - GPU will be available in container"; \
		echo "✓ Using USER_ID=$$USER_ID, GROUP_ID=$$GROUP_ID for file permissions"; \
		USER_ID=$$USER_ID GROUP_ID=$$GROUP_ID docker compose up -d --build; \
	else \
		echo "✓ No GPU detected - running in CPU mode"; \
		echo "✓ Using USER_ID=$$USER_ID, GROUP_ID=$$GROUP_ID for file permissions"; \
		USER_ID=$$USER_ID GROUP_ID=$$GROUP_ID docker compose up -d --build; \
	fi
	@echo "Development environment started. Use 'make shell' to enter."

down:
	@# Stop all containers including hierarchical profile services
	docker compose --profile hierarchical down

down-dev:
	@# Stop only the main development container (preserves Ollama if running)
	docker compose down

restart:
	docker compose restart

# Start development environment with Ollama server for LLM inference
up-hierarchical:
	@echo "Starting development environment with Ollama server..."
	@if [ -z "$$USER_ID" ]; then export USER_ID=$$(id -u); fi; \
	if [ -z "$$GROUP_ID" ]; then export GROUP_ID=$$(id -g); fi; \
	echo "✓ Using USER_ID=$$USER_ID, GROUP_ID=$$GROUP_ID for file permissions"; \
	USER_ID=$$USER_ID GROUP_ID=$$GROUP_ID HIERARCHICAL_MODE=true \
		docker compose --profile hierarchical up -d --build
	@echo "Development environment with Ollama started."
	@echo "Use 'make shell' to enter the dev container."
	@echo "Ollama API available at http://localhost:11434"
	@echo "Pull a model with: docker exec ollama ollama pull llama2"

shell:
	docker compose exec empo-dev bash

logs:
	docker compose logs -f

# Training commands
train:
	docker compose exec empo-dev python train.py --num-episodes 100

example:
	docker compose exec empo-dev python examples/simple_example.py

# Development commands
test:
	@echo "Running tests..."
	docker compose exec empo-dev pytest tests/ -v

lint:
	@echo "Running linters..."
	docker compose exec empo-dev ruff check .
	docker compose exec empo-dev black --check .

# Cleanup
clean:
	@echo "Cleaning up outputs and cache..."
	rm -rf outputs/ logs/ __pycache__/ .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# GPU Docker image build
build-gpu:
	@echo "Building GPU-enabled Docker image for cluster..."
	@echo "Image: $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"
	docker build -f Dockerfile.gpu \
		-t $(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG) \
		-t $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG) \
		.
	@echo "✓ GPU image built successfully"

# Push GPU image to Docker Hub
push-gpu:
	@echo "Pushing GPU image to $(DOCKER_REGISTRY)..."
	@echo "Image: $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"
	@if [ "$(DOCKER_USERNAME)" = "$(shell whoami)" ]; then \
		echo ""; \
		echo "WARNING: DOCKER_USERNAME not set, using system username: $(DOCKER_USERNAME)"; \
		echo "Set DOCKER_USERNAME in .env or environment to use your Docker Hub username"; \
		echo ""; \
	fi
	docker push $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)
	@echo "✓ GPU image pushed successfully"
	@echo ""
	@echo "On cluster, pull with:"
	@echo "  apptainer pull $(SIF_FILE) docker://$(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"

# Build SIF file from GPU Docker image
build-sif:
	@echo "Converting GPU Docker image to Singularity SIF file..."
	@echo "Output: $(SIF_FILE)"
	@if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then \
		echo ""; \
		echo "ERROR: Neither apptainer nor singularity found!"; \
		echo "This target requires Apptainer/Singularity to be installed."; \
		echo ""; \
		echo "Alternatives:"; \
		echo "  1. Use 'make up-gpu-docker-hub' to push to Docker Hub instead"; \
		echo "  2. Install Apptainer: https://apptainer.org/docs/admin/main/installation.html"; \
		echo "  3. Use Docker Desktop with 'docker save' and convert on cluster"; \
		echo ""; \
		exit 1; \
	fi
	@if command -v apptainer &> /dev/null; then \
		apptainer build $(SIF_FILE) docker-daemon://$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG); \
	else \
		singularity build $(SIF_FILE) docker-daemon://$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG); \
	fi
	@echo "✓ SIF file created: $(SIF_FILE)"
	@echo ""
	@echo "Copy to cluster with:"
	@echo "  scp $(SIF_FILE) user@cluster:~/bega/empo/"
	@echo ""
	@echo "On cluster, run with:"
	@echo "  cd ~/bega/empo/git"
	@echo "  sbatch ../scripts/run_cluster_sif.sh"

# Build and push GPU image to Docker Hub (no Singularity needed locally)
up-gpu-docker-hub: build-gpu push-gpu
	@echo ""
	@echo "==================================="
	@echo "GPU image ready on Docker Hub!"
	@echo "==================================="
	@echo ""
	@echo "On cluster, pull and run with:"
	@echo "  cd ~/bega/empo"
	@echo "  mkdir -p git"
	@echo "  cd git && git clone <your-repo-url> . && cd .."
	@echo "  apptainer pull empo.sif docker://$(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)"
	@echo "  cd git && sbatch ../scripts/run_cluster_sif.sh"

# Build GPU image and convert to SIF file locally
up-gpu-sif-file: build-gpu build-sif
	@echo ""
	@echo "==================================="
	@echo "GPU SIF file ready for cluster!"
	@echo "==================================="
	@echo ""
	@echo "Copy to cluster and run:"
	@echo "  scp $(SIF_FILE) user@cluster:~/bega/empo/"
	@echo "  ssh user@cluster"
	@echo "  cd ~/bega/empo/git"
	@echo "  sbatch ../scripts/run_cluster_sif.sh"

# Build Docker image with hierarchical dependencies (Ollama, MineLand)
# These are large packages that require Java JDK 17 and Node.js 18
build-hierarchical:
	@echo "Building Docker image with hierarchical dependencies..."
	docker build --build-arg DEV_MODE=true --build-arg HIERARCHICAL_MODE=true \
		-t $(DOCKER_IMAGE_NAME):hierarchical .
	@echo "✓ Hierarchical image built successfully"

# Build GPU Docker image with hierarchical dependencies
build-gpu-hierarchical:
	@echo "Building GPU Docker image with hierarchical dependencies..."
	@echo "Image: $(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)-hierarchical"
	docker build -f Dockerfile.gpu \
		--build-arg HIERARCHICAL_MODE=true \
		-t $(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)-hierarchical \
		-t $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(GPU_IMAGE_TAG)-hierarchical \
		.
	@echo "✓ GPU hierarchical image built successfully"

# Test MineLand installation (requires hierarchical build)
test-mineland:
	@echo "Testing MineLand installation (basic import tests)..."
	docker compose exec empo-dev python tests/test_mineland_installation.py

# Test MineLand + Ollama integration (requires up-hierarchical and qwen2.5-vl:3b model)
test-mineland-integration:
	@echo "Testing MineLand + Ollama integration..."
	@echo "Make sure you have:"
	@echo "  1. Started with: make up-hierarchical"
	@echo "  2. Pulled model: docker exec ollama ollama pull qwen2.5-vl:3b"
	@echo ""
	docker compose exec empo-dev python tests/test_mineland_installation.py --integration
