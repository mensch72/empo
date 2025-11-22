.PHONY: help build build-cpu up up-cpu up-gpu down restart shell logs clean test lint

# Default target
help:
	@echo "EMPO Development Commands"
	@echo "========================="
	@echo "make build          - Build Docker image (with CUDA, ~5GB)"
	@echo "make build-cpu      - Build CPU-only Docker image (lighter, ~2GB)"
	@echo "make up             - Start development environment (with CUDA base)"
	@echo "make up-cpu         - Start development environment (CPU-only, lighter)"
	@echo "make up-gpu         - Start development environment (GPU mode)"
	@echo "make down           - Stop development environment"
	@echo "make restart        - Restart development environment"
	@echo "make shell          - Open shell in container"
	@echo "make logs           - Show container logs"
	@echo "make train          - Run training script"
	@echo "make example        - Run simple example"
	@echo "make test           - Run tests (when implemented)"
	@echo "make lint           - Run linters (when implemented)"
	@echo "make clean          - Clean up outputs and cache"
	@echo "make cluster-image  - Build Singularity/Apptainer image"

# Docker Compose commands
build:
	docker compose build

build-cpu:
	docker compose -f docker-compose.cpu.yml build

up:
	docker compose up -d --build
	@echo "Development environment started (CUDA base image)."
	@echo "Use 'make shell' to enter."
	@echo "For lighter CPU-only image, use 'make up-cpu' instead."
	@echo "For GPU support, use 'make up-gpu' instead."

up-cpu:
	docker compose -f docker-compose.cpu.yml up -d --build
	@echo "Development environment started (CPU-only, lighter image)."
	@echo "Use 'make shell' to enter."

up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
	@echo "Development environment started (GPU mode)."
	@echo "Use 'make shell' to enter."

down:
	docker compose down

restart:
	docker compose restart

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

# Cluster image (requires Docker to Singularity conversion)
cluster-image:
	@echo "Building production Docker image..."
	docker build -t empo:latest .
	@echo ""
	@echo "To convert to Singularity/Apptainer image:"
	@echo "  1. Push to registry: docker push youruser/empo:latest"
	@echo "  2. On cluster: apptainer pull empo.sif docker://youruser/empo:latest"
	@echo "  Or see scripts/setup_cluster_image.sh for more options"
