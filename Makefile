.PHONY: help build up down restart shell logs clean test lint

# Default target
help:
	@echo "EMPO Development Commands"
	@echo "========================="
	@echo "make build          - Build Docker image"
	@echo "make up             - Start development environment (auto-detects GPU)"
	@echo "make down           - Stop development environment"
	@echo "make restart        - Restart development environment"
	@echo "make shell          - Open shell in container"
	@echo "make logs           - Show container logs"
	@echo "make train          - Run training script"
	@echo "make example        - Run simple example"
	@echo "make test           - Run tests"
	@echo "make lint           - Run linters"
	@echo "make clean          - Clean up outputs and cache"

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
