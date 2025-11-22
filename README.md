# EMPO: Empowerment-based Multi-Agent Reinforcement Learning

A unified framework for training multi-agent reinforcement learning (MARL) agents using empowerment-based objectives on multigrid environments. This repository supports both local development (Docker Compose) and cluster deployment (Singularity/Apptainer).

## Features

- 🚀 Unified Docker image for development and cluster deployment
- 🎮 MARL training with empowerment-based objectives
- 🔧 Easy local development with Docker Compose
- 🖥️ Cluster-ready with Singularity/Apptainer support
- 🎯 GPU acceleration support (NVIDIA CUDA)
- 📊 Integration with TensorBoard and Weights & Biases

## Quick Start

### Prerequisites

**For Local Development:**
- Docker Engine 20.10+ with Docker Compose v2
- NVIDIA Docker runtime (for GPU support)
- NVIDIA drivers (for GPU support)

**For Cluster Deployment:**
- Singularity/Apptainer 1.0+
- SLURM or similar job scheduler (optional)

### Installation

Clone the repository:

```bash
git clone https://github.com/pik-gane/empo.git
cd empo
```

## Local Development

### 1. Build and Start the Development Environment

```bash
# Single command that works everywhere
make up

# Or using docker compose directly
docker compose up -d
```

The setup automatically:
- Uses a lightweight Ubuntu-based image (~2GB)
- Detects if you have an NVIDIA GPU
- Shows you whether GPU is available or running in CPU mode
- No CUDA libraries downloaded unless needed for cluster deployment
- **Caches apt and pip packages** for much faster rebuilds
  - First build: ~5-10 minutes (downloads packages)
  - Subsequent builds: ~30 seconds (uses cached packages)
  - Only rebuilds changed layers (e.g., when requirements.txt changes)

### 2. Enter the Container

```bash
# Attach to the running container
docker compose exec empo-dev bash

# Or use docker exec
docker exec -it empo-dev bash
```

### 3. Run Training

Inside the container:

```bash
# Run the example training script
python train.py --num-episodes 100

# Or with custom arguments
python train.py \
  --env-name CartPole-v1 \
  --num-episodes 1000 \
  --lr 0.001 \
  --output-dir ./outputs
```

### 4. Development Workflow

The repository is bind-mounted at `/workspace`, so any changes you make locally are immediately reflected in the container:

```bash
# Edit files on your host machine with your favorite editor
vim train.py

# Changes are immediately available in the container
docker compose exec empo-dev python train.py
```

### 5. GPU Support

GPU support is automatically detected when you run `make up`:

```bash
# Just use the standard command
make up
```

The system will:
- Detect if you have an NVIDIA GPU with `nvidia-smi`
- Display "GPU detected" or "No GPU detected"  
- Work correctly either way - no configuration needed

**Verifying GPU access (if GPU is available):**

```bash
# This will work if GPU was detected
docker compose exec empo-dev nvidia-smi
docker compose exec empo-dev python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note:** The Docker image uses a lightweight Ubuntu base (~2GB), not CUDA base, so it's fast to download on any system. PyTorch automatically uses GPU if available, or CPU otherwise.

### 6. Jupyter Notebook (Optional)

```bash
# Start Jupyter inside the container
docker compose exec empo-dev jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Access at http://localhost:8888
```

### 7. TensorBoard (Optional)

The training script automatically logs metrics to TensorBoard. To view them:

```bash
# Start TensorBoard (from within the container)
docker compose exec empo-dev tensorboard --logdir=./outputs --host=0.0.0.0

# Or from your host machine (if you have tensorboard installed)
tensorboard --logdir=./outputs

# Access at http://localhost:6006
```

The training script writes metrics like episode rewards, episode lengths, and learning rates to TensorBoard. Even in demo mode (without a real environment), it logs sample data so you can verify TensorBoard is working correctly.

### 8. Stop the Environment

```bash
# Stop the container
docker compose down

# Stop and remove volumes
docker compose down -v
```

## Cluster Deployment

### 1. Build the Docker Image

First, build the production Docker image (without dev dependencies):

```bash
# Build production image
docker build -t empo:latest .

# Or build with a specific tag
docker build -t empo:v0.1.0 .
```

### 2. Convert to Singularity/Apptainer Image

There are several ways to get the image on your cluster:

#### Option A: Pull from a Registry (Recommended)

```bash
# Push to Docker Hub or GitHub Container Registry
docker tag empo:latest yourusername/empo:latest
docker push yourusername/empo:latest

# On the cluster, pull and convert to SIF format
apptainer pull empo.sif docker://yourusername/empo:latest
```

#### Option B: Build Directly from Dockerfile

```bash
# On the cluster with Apptainer installed
apptainer build empo.sif Dockerfile
```

#### Option C: Transfer from Local Docker

```bash
# Save Docker image to a tar file
docker save empo:latest -o empo.tar

# Transfer to cluster (e.g., via scp)
scp empo.tar cluster:/path/to/destination/

# On the cluster, load and convert
apptainer build empo.sif docker-archive://empo.tar
```

### 3. Test the Singularity Image

```bash
# Test basic functionality
apptainer exec empo.sif python3 --version

# Test with GPU support
apptainer exec --nv empo.sif python3 -c "import torch; print(torch.cuda.is_available())"

# Run the training script
apptainer exec --nv -B $(pwd):/workspace empo.sif python /workspace/train.py --num-episodes 10
```

### 4. Submit a SLURM Job

Edit the provided SLURM script and submit:

```bash
# Create logs directory
mkdir -p logs

# Edit the script with your parameters
vim scripts/run_cluster.sh

# Submit the job
sbatch scripts/run_cluster.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/empo_<job_id>.out
```

### 5. Interactive Cluster Session

For interactive development on the cluster:

```bash
# Request an interactive GPU node
srun --partition=gpu --gres=gpu:1 --mem=32G --time=4:00:00 --pty bash

# Run commands interactively
apptainer shell --nv -B $(pwd):/workspace empo.sif
python /workspace/train.py --num-episodes 100
```

## Project Structure

```
empo/
├── Dockerfile                 # Unified Docker image definition
├── docker-compose.yml         # Local development setup
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── train.py                   # Main training script
├── src/
│   └── empo/                  # Main package
│       └── __init__.py
├── vendor/
│   └── multigrid/             # Vendored Multigrid source (live editable)
├── configs/
│   └── default.yaml           # Example configuration
├── scripts/
│   ├── run_cluster.sh         # SLURM job script
│   └── setup_cluster_image.sh # Cluster image setup helper
├── examples/                  # Example scripts and notebooks
├── VENDOR.md                  # Documentation for vendored dependencies
└── README.md                  # This file
```

## Vendored Dependencies

This repository includes the [Multigrid](https://github.com/ArnaudFickinger/gym-multigrid) source code in `vendor/multigrid/` to enable live editing without container rebuilds.

**How it works:**
- Multigrid is imported via `PYTHONPATH` (not pip installed)
- Edit files in `vendor/multigrid/gym_multigrid/` and changes take effect immediately
- No Docker rebuild needed for modifications
- Perfect for making extensive changes to environments

**Modifying Multigrid:**
```bash
# 1. Edit source files
vim vendor/multigrid/gym_multigrid/envs/collect_game.py

# 2. Restart Python or re-import (no rebuild needed)
docker compose restart empo-dev
```

**Updating from upstream:**
```bash
git subtree pull --prefix=vendor/multigrid https://github.com/ArnaudFickinger/gym-multigrid.git master --squash
```

See [VENDOR.md](VENDOR.md) for detailed documentation on managing vendored dependencies.

## Configuration

Training can be configured via command-line arguments or YAML configuration files:

```bash
# Using command-line arguments
python train.py --env-name CartPole-v1 --num-episodes 1000 --lr 0.001

# Using a config file (implement config loading in your code)
python train.py --config configs/default.yaml
```

## Environment Variables

### Docker Compose

Set these in a `.env` file or export before running:

```bash
# User ID mapping (for file permissions)
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs

# Weights & Biases
export WANDB_API_KEY=your_key_here
```

### Cluster

Set these in your job script or environment:

```bash
export REPO_PATH=/path/to/empo
export IMAGE_PATH=/path/to/empo.sif
export SCRIPT_PATH=train.py
```

## Troubleshooting

### Docker Compose Issues

**GPU not detected:**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker Compose GPU syntax
docker compose config
```

**Permission issues:**

The `make up` command automatically sets USER_ID and GROUP_ID to match your host user. If you encounter permission issues:

```bash
# Make sure you're using make up (recommended)
make up

# Or manually set user IDs with docker compose
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
docker compose up --build
```

If you still have issues, ensure you have write permissions to the repository directory on your host system.

### Singularity/Apptainer Issues

**GPU not available:**
```bash
# Ensure --nv flag is used
apptainer exec --nv empo.sif nvidia-smi

# Check CUDA libraries
apptainer exec --nv empo.sif python -c "import torch; print(torch.version.cuda)"
```

**Mount point issues:**
```bash
# Ensure bind mount paths exist and are accessible
apptainer exec -B /full/path/to/repo:/workspace empo.sif ls /workspace
```

**Image building issues:**
```bash
# Use --fakeroot if you don't have root privileges
apptainer build --fakeroot empo.sif Dockerfile
```

## Advanced Usage

### Custom Dependencies

Edit `requirements.txt` to add your dependencies:

```bash
# Add to requirements.txt
your-package>=1.0.0

# Rebuild the image
docker compose up --build
```

### Multi-GPU Training

```bash
# Docker Compose (use specific GPUs)
CUDA_VISIBLE_DEVICES=0,1 docker compose up

# Cluster (request multiple GPUs)
#SBATCH --gres=gpu:2
```

### Distributed Training with MPI

```bash
# In the container or on the cluster
mpirun -np 4 python train.py --distributed
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both Docker and Singularity
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Documentation

- **[README.md](README.md)** - This file, comprehensive setup and usage guide
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Detailed implementation notes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[VENDOR.md](VENDOR.md)** - Managing vendored dependencies (Multigrid)
- **[.env.example](.env.example)** - Environment variables template

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on PyTorch and Gymnasium
- Supports PettingZoo and Multigrid environments
- Inspired by empowerment-driven intrinsic motivation research

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and discussions
- Refer to the troubleshooting section above

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{empo2024,
  title = {EMPO: Empowerment-based Multi-Agent Reinforcement Learning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/pik-gane/empo}
}
```
