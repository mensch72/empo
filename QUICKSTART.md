# Quick Start Guide

Get up and running with EMPO in under 5 minutes!

## Prerequisites Check

```bash
# Verify Docker is installed
docker --version

# Verify Docker Compose is installed
docker compose version

# Optional: Check for GPU support
nvidia-smi
```

## Local Development (Docker Compose)

### 1. Clone and Setup

```bash
git clone https://github.com/pik-gane/empo.git
cd empo

# Verify setup
bash scripts/verify_setup.sh
```

### 2. Start Development Environment

**CPU mode (default - works on all systems):**
```bash
# Start the container
make up
# Or manually
docker compose up -d
```

**GPU mode (for systems with NVIDIA GPU):**
```bash
# Start with GPU support
make up-gpu
# Or manually
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### 3. Enter the Container

```bash
# Using make
make shell

# Or using docker compose
docker compose exec empo-dev bash
```

### 4. Run Your First Training

Inside the container:

```bash
# Run example training
python train.py --num-episodes 100

# Run simple example
python examples/simple_example.py
```

### 5. View Results

```bash
# Check outputs
ls -la outputs/

# For TensorBoard (if implemented)
tensorboard --logdir=outputs --host=0.0.0.0
# Access at http://localhost:6006
```

### 6. Stop Environment

```bash
make down
# Or: docker compose down
```

## Cluster Deployment (Singularity/Apptainer)

### 1. Build Docker Image Locally

```bash
docker build -t empo:latest .
```

### 2. Push to Registry

```bash
# Login to Docker Hub (or GHCR)
docker login

# Tag and push
docker tag empo:latest yourusername/empo:latest
docker push yourusername/empo:latest
```

### 3. On the Cluster

```bash
# Pull and convert to Singularity
apptainer pull empo.sif docker://yourusername/empo:latest

# Test
apptainer exec empo.sif python3 --version

# Test with GPU
apptainer exec --nv empo.sif nvidia-smi
```

### 4. Run Training

```bash
# Interactive
apptainer exec --nv -B $(pwd):/workspace empo.sif \
  python /workspace/train.py --num-episodes 1000

# Submit SLURM job
sbatch scripts/run_cluster.sh
```

## Common Commands

### Development

```bash
make help          # Show all commands
make build         # Build image
make up            # Start container
make down          # Stop container
make shell         # Open shell
make train         # Run training
make example       # Run example
make logs          # Show logs
make clean         # Clean outputs
```

### Docker Compose

```bash
docker compose up -d              # Start in background
docker compose down               # Stop
docker compose logs -f            # Follow logs
docker compose exec empo-dev bash # Enter container
docker compose restart            # Restart
```

### Manual Docker

```bash
# Build
docker build -t empo:latest .

# Run interactive
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  empo:latest bash

# Run training
docker run --rm \
  --gpus all \
  -v $(pwd):/workspace \
  empo:latest \
  python /workspace/train.py
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# For Docker Compose, ensure nvidia runtime is default
# Edit /etc/docker/daemon.json:
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Restart Docker
sudo systemctl restart docker
```

### Permission Issues

```bash
# Set user IDs
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

# Rebuild
make down && make build && make up
```

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8888

# Change ports in docker-compose.yml or stop conflicting service
```

### Cluster Issues

```bash
# Check Singularity version
apptainer --version

# Test basic functionality
apptainer exec empo.sif cat /etc/os-release

# Check GPU binding
apptainer exec --nv empo.sif python3 -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

- Read the full [README.md](README.md)
- Explore [configs/default.yaml](configs/default.yaml)
- Check example scripts in [examples/](examples/)
- Customize for your research needs

## Support

- GitHub Issues: https://github.com/pik-gane/empo/issues
- Documentation: See README.md
