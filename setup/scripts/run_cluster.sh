#!/bin/bash
# Singularity/Apptainer job submission script for SLURM clusters
#
# Usage: sbatch scripts/run_cluster.sh
# Or modify and adapt for your specific cluster setup
#
# Security Note: This script uses environment variables for configuration.
# Ensure variables are properly validated in production environments.

#SBATCH --job-name=empo-training
#SBATCH --output=logs/empo_%j.out
#SBATCH --error=logs/empo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Configuration - modify these for your setup
# Use absolute paths for security and clarity
REPO_PATH="${REPO_PATH:-$(pwd)}"
IMAGE_PATH="${IMAGE_PATH:-./empo.sif}"
SCRIPT_PATH="${SCRIPT_PATH:-train.py}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"

echo "==================================="
echo "EMPO Cluster Training Job"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "==================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU information
nvidia-smi

echo ""
echo "Starting training with Apptainer..."
echo "Repository: $REPO_PATH"
echo "Image: $IMAGE_PATH"
echo "Script: $SCRIPT_PATH"
echo "Episodes: $NUM_EPISODES"
echo ""

# Run the training script using Apptainer with GPU support
# --nv: Enable NVIDIA GPU support
# -B: Bind mount the repository into the container
# --pwd: Set working directory inside container (fixes the chdir warning)
apptainer exec --nv \
  --pwd /workspace \
  -B "${REPO_PATH}:/workspace" \
  "${IMAGE_PATH}" \
  python /workspace/"${SCRIPT_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --output-dir "${OUTPUT_DIR}"

echo ""
echo "==================================="
echo "Job completed: $(date)"
echo "==================================="
