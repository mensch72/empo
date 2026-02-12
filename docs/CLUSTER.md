# Cluster Deployment Guide

This guide explains how to deploy EMPO on HPC clusters using Singularity/Apptainer with GPU support.

## Overview

EMPO provides two methods for cluster deployment:

1. **Docker Hub method** (`make up-gpu-docker-hub`) - Build locally, push to Docker Hub, pull on cluster
2. **SIF file method** (`make up-gpu-sif-file`) - Build locally, transfer SIF file directly to cluster

Both methods use a GPU-enabled Docker image (`Dockerfile.gpu`) with CUDA 12.1 support.

## Prerequisites

### Local Machine
- Docker installed and running
- (Optional) Singularity/Apptainer for SIF file method
- Docker Hub account for Docker Hub method

### Cluster
- Singularity or Apptainer installed
- NVIDIA GPUs with drivers
- SLURM job scheduler (or adapt scripts for your scheduler)

## Method 1: Docker Hub (Recommended)

This method doesn't require Singularity locally, making it easier for most users.

### Step 1: Configure Docker Hub credentials

Create or edit `.env`:
```bash
cp .env.example .env
```

Edit `.env` and set:
```bash
DOCKER_REGISTRY=docker.io
DOCKER_USERNAME=your-docker-hub-username
GPU_IMAGE_TAG=gpu-latest
```

### Step 2: Build and push GPU image

```bash
make up-gpu-docker-hub
```

This will:
- Build a GPU-enabled Docker image using `Dockerfile.gpu`
- Tag it as `docker.io/your-docker-hub-username/empo:gpu-latest`
- Push it to Docker Hub
- Display instructions for pulling on the cluster

**Note:** You need to be logged in to Docker Hub:
```bash
docker login
```

### Step 3: On the cluster

```bash
# Navigate to your workspace
cd ~/bega/empo

# Clone your repository
mkdir -p git
cd git
git clone https://github.com/yourusername/empo.git .
cd ..

# Pull the GPU image from Docker Hub
apptainer pull empo.sif docker://your-docker-hub-username/empo:gpu-latest

# Submit training job
cd git
sbatch ../setup/scripts/run_cluster_sif.sh
```

## Method 2: SIF File (Direct Transfer)

This method builds the SIF file locally and transfers it directly to the cluster, bypassing Docker Hub.

### Step 1: Install Singularity/Apptainer locally

- **Ubuntu/Debian:** Follow [Apptainer installation guide](https://apptainer.org/docs/admin/main/installation.html)
- **macOS:** Not directly supported; use Method 1 instead

### Step 2: Build SIF file locally

```bash
make up-gpu-sif-file
```

This will:
- Build a GPU-enabled Docker image using `Dockerfile.gpu`
- Convert it to a Singularity SIF file (`empo-gpu.sif`)
- Display instructions for transferring to the cluster

### Step 3: Transfer to cluster

```bash
scp empo-gpu.sif user@cluster.example.com:~/bega/empo/
```

### Step 4: On the cluster

```bash
# Clone your repository (if not already done)
cd ~/bega/empo
mkdir -p git
cd git
git clone https://github.com/yourusername/empo.git .
cd ..

# Submit training job (SIF file is in parent directory)
cd git
sbatch ../setup/scripts/run_cluster_sif.sh
```

## Cluster Job Scripts

### Using run_cluster_sif.sh (Recommended)

This script is optimized for the directory structure where:
- SIF file is in `~/bega/empo/empo.sif` or `~/bega/empo/empo-gpu.sif`
- Repository is cloned in `~/bega/empo/git/`

```bash
cd ~/bega/empo/git
sbatch ../setup/scripts/run_cluster_sif.sh
```

Features:
- Fixes working directory warnings with `--pwd /workspace`
- Configurable via environment variables
- Default: trains for 1000 episodes

Customize with environment variables:
```bash
REPO_PATH=~/bega/empo/git \
IMAGE_PATH=~/bega/empo/empo-gpu.sif \
NUM_EPISODES=5000 \
sbatch ../setup/scripts/run_cluster_sif.sh
```

### Using run_cluster.sh (Generic)

For flexible deployment scenarios:

```bash
sbatch setup/scripts/run_cluster.sh
```

## Testing GPU Access

Test that the cluster can access GPUs through the container:

```bash
# Test NVIDIA driver
apptainer exec --nv empo.sif nvidia-smi

# Test PyTorch CUDA
apptainer exec --nv --pwd /workspace -B ~/bega/empo/git:/workspace empo.sif \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### "Error changing the container working directory"

This is fixed by using `--pwd /workspace` in the apptainer command. The updated scripts include this flag.

If you see this warning, update your commands:
```bash
# OLD (warning)
apptainer exec empo.sif python train.py

# NEW (no warning)
apptainer exec --pwd /workspace -B $(pwd):/workspace empo.sif python /workspace/train.py
```

### "can't open file"

This happens when the working directory is wrong. Always:
1. Use `--pwd /workspace` to set the container working directory
2. Use `-B /path/to/repo:/workspace` to bind mount your repository
3. Use absolute paths: `python /workspace/train.py`

### File permissions

The GPU image uses UID/GID 1000 by default. If your cluster user has a different UID:

```bash
# Check your UID
id -u

# Rebuild with custom UID (requires rebuilding locally)
docker build -f Dockerfile.gpu --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t empo:gpu-latest .
```

### CUDA version mismatch

The GPU image uses CUDA 12.1. If your cluster has a different CUDA version:

1. Check cluster CUDA version: `nvidia-smi`
2. Edit `Dockerfile.gpu` to match: `FROM nvidia/cuda:XX.X.X-cudnn8-runtime-ubuntu22.04`
3. Rebuild and redeploy

## Directory Structure

Expected cluster layout:
```
~/bega/empo/
├── empo.sif                    # Singularity image
├── empo-gpu.sif               # (or this name)
├── git/                       # Your repository clone
│   ├── train.py
│   ├── examples/
│   ├── setup/scripts/
│   │   └── run_cluster_sif.sh
│   ├── outputs/               # Created by training
│   └── logs/                  # SLURM logs
└── setup/scripts/                   # (optional) copy of scripts
```

## Updating the Image

### Docker Hub method:
```bash
# On local machine
cd /path/to/empo
git pull
make up-gpu-docker-hub

# On cluster
cd ~/bega/empo
apptainer pull empo.sif docker://your-docker-hub-username/empo:gpu-latest
```

### SIF file method:
```bash
# On local machine
cd /path/to/empo
git pull
make up-gpu-sif-file
scp empo-gpu.sif user@cluster:~/bega/empo/
```

## Advanced Configuration

### Custom job parameters

Edit `setup/scripts/run_cluster_sif.sh` for your cluster:

```bash
#SBATCH --partition=gpu          # Your GPU partition name
#SBATCH --gres=gpu:1             # Number of GPUs
#SBATCH --cpus-per-task=4        # CPU cores
#SBATCH --mem=32G                # Memory
#SBATCH --time=24:00:00          # Walltime
```

### Multiple GPU training

Not yet implemented, but to prepare:

```bash
#SBATCH --gres=gpu:4             # Request 4 GPUs
```

Then modify your training script to use `torch.nn.DataParallel` or `DistributedDataParallel`.

## Best Practices

1. **Test locally first**: Always test with `make up` before deploying to cluster
2. **Start small**: Run a short job (100 episodes) to verify everything works
3. **Monitor resources**: Check `logs/empo_*.out` for GPU usage and performance
4. **Version control**: Tag your releases before pushing to Docker Hub
5. **Clean up**: Remove old SIF files to save disk space on cluster

## Support

If you encounter issues:
1. Check SLURM logs: `cat logs/empo_*.err`
2. Test the SIF file interactively: `apptainer shell --nv --pwd /workspace -B ~/bega/empo/git:/workspace empo.sif`
3. Verify GPU access: `apptainer exec --nv empo.sif nvidia-smi`
4. Check repository issues: https://github.com/pik-gane/empo/issues
