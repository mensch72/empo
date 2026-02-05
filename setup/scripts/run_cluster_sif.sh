#!/bin/bash
# SLURM job script for running with locally-copied SIF file
# This script fixes the working directory issue with Singularity/Apptainer
#
# Usage: 
#   1. Copy SIF to cluster: scp empo.sif user@cluster:~/bega/empo/
#   2. Submit job: sbatch scripts/run_cluster_sif.sh

#SBATCH --job-name=empo-training
#SBATCH --output=logs/empo_%j.out
#SBATCH --error=logs/empo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Configuration - modify these for your setup
REPO_PATH="${REPO_PATH:-$(pwd)/git}"
IMAGE_PATH="${IMAGE_PATH:-$(pwd)/empo.sif}"
SCRIPT_PATH="${SCRIPT_PATH:-train.py}"
NUM_EPISODES="${NUM_EPISODES:-1000}"

echo "==================================="
echo "EMPO Cluster Training Job"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Current directory: $(pwd)"
echo "==================================="

# Create logs directory in repo if it doesn't exist
mkdir -p "${REPO_PATH}/logs"

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
  --output-dir /workspace/outputs

echo ""
echo "==================================="
echo "Job completed: $(date)"
echo "==================================="
