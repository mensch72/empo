#!/bin/bash
# Script to pull Docker image and convert to Singularity/Apptainer format
#
# Usage: ./scripts/setup_cluster_image.sh [IMAGE_NAME]

set -e

# Configuration
DOCKER_IMAGE="${1:-empo:latest}"
OUTPUT_SIF="${2:-empo.sif}"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"

echo "==================================="
echo "Singularity/Apptainer Image Setup"
echo "==================================="
echo "Docker image: $DOCKER_IMAGE"
echo "Output SIF: $OUTPUT_SIF"
echo ""

# Check if apptainer or singularity is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "Error: Neither apptainer nor singularity found!"
    echo "Please install Apptainer/Singularity on your cluster."
    exit 1
fi

echo "Using container runtime: $CONTAINER_CMD"
echo ""

# Option 1: Pull from a registry (if image is pushed to DockerHub/GHCR)
echo "To pull from a registry, the image must be pushed first:"
echo "  docker tag $DOCKER_IMAGE ${REGISTRY}/youruser/$DOCKER_IMAGE"
echo "  docker push ${REGISTRY}/youruser/$DOCKER_IMAGE"
echo "  $CONTAINER_CMD pull $OUTPUT_SIF docker://${REGISTRY}/youruser/$DOCKER_IMAGE"
echo ""

# Option 2: Build from local Docker daemon (if available on cluster)
echo "To build from local Docker daemon:"
echo "  $CONTAINER_CMD build $OUTPUT_SIF docker-daemon://$DOCKER_IMAGE"
echo ""

# Option 3: Build directly from Dockerfile
echo "To build directly from Dockerfile:"
echo "  $CONTAINER_CMD build $OUTPUT_SIF Dockerfile"
echo ""

echo "==================================="
echo "Testing the image"
echo "==================================="
echo ""

if [ -f "$OUTPUT_SIF" ]; then
    echo "Image found at $OUTPUT_SIF"
    echo "Testing basic functionality..."
    $CONTAINER_CMD exec "$OUTPUT_SIF" python3 --version
    $CONTAINER_CMD exec "$OUTPUT_SIF" python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "Testing GPU support..."
        $CONTAINER_CMD exec --nv "$OUTPUT_SIF" python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    fi
    
    echo ""
    echo "✓ Image is ready for cluster use!"
else
    echo "No image found. Please build or pull the image first using one of the methods above."
fi
