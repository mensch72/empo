# Quick Start Guide

Get up and running with EMPO in under 5 minutes!

## Google Colab (Quickest Option)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mensch72/empo/blob/main/notebooks/colab_launcher.ipynb)

**No installation required!** Click the badge above to run EMPO directly in your browser.

For manual Colab setup:

```python
# Clone repository
!git clone --depth 1 https://github.com/mensch72/empo.git
%cd empo

# Install dependencies
!apt-get update -qq && apt-get install -qq graphviz > /dev/null 2>&1
!pip install -q -r setup/requirements-colab.txt

# Configure Python path
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'vendor', 'multigrid'))

# You're ready!
from empo import WorldModel, PossibleGoal
```

## Kaggle (Best Free GPU Option)

Kaggle offers **30 hours/week of free GPU** with reliable background execution.

### Using the Launcher Notebook

1. Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. Enable **Internet** (Settings → Internet → On)
3. Enable **GPU** (Settings → Accelerator → GPU T4 x2)
4. Copy these cells:

```python
# Cell 1: Clone repository
!git clone --depth 1 https://github.com/mensch72/empo.git
%cd empo
```

```python
# Cell 2: Setup
%run setup/scripts/kaggle_setup.py
```

```python
# Cell 3: Run any example script
%run examples/phase2/phase2_robot_policy_demo.py --quick
```

### Background Execution

For long training runs (30+ minutes):
1. Edit cell 3 with your desired script
2. Click **"Save Version"** → **"Save & Run All (Commit)"**
3. Close browser - Kaggle continues running
4. Download outputs from the **"Output"** tab when complete

### Available Example Scripts

```python
# Quick demos (< 1 min)
%run examples/multigrid/simple_example.py
%run examples/multigrid/state_management_demo.py

# Medium demos (1-5 min)
%run examples/phase1/human_policy_prior_example.py --quick
%run examples/phase1/neural_policy_prior_demo.py --quick

# Phase 2 training (30-60+ min)
%run examples/phase2/phase2_robot_policy_demo.py              # Full training
%run examples/phase2/phase2_robot_policy_demo.py --quick      # Quick test
%run examples/phase2/phase2_robot_policy_demo.py --ensemble   # Random environments
%run examples/phase2/phase2_robot_policy_demo.py --tabular    # Lookup tables
```

See [examples/README.md](examples/README.md) for all available scripts and flags.

### Platform Comparison

| Feature | Kaggle | Colab Free |
|---------|--------|------------|
| GPU quota | 30 hrs/week (fixed) | ~4-8 hrs/day (dynamic) |
| Background execution | ✅ Yes | ❌ No |
| Session limit | 12 hours | 12 hours |
| Idle timeout | Ignored in background | 90 min disconnect |

**Note**: Neither platform supports `--async` mode (multiprocessing doesn't work in notebooks).

## Prerequisites Check (for Local Development)

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
bash setup/scripts/verify_setup.sh
```

### 2. Start Development Environment

```bash
# Single command that auto-detects GPU
make up
```

That's it! The setup automatically detects if you have a GPU and configures accordingly.

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
python examples/multigrid/simple_example.py
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
sbatch setup/scripts/run_cluster.sh
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
- Check example scripts in [examples/](examples/)
- Customize for your research needs

## Support

- GitHub Issues: https://github.com/pik-gane/empo/issues
- Documentation: See README.md
