# EMPO Setup Directory

This directory contains all setup and configuration files for the EMPO project.

## Directory Structure

```
setup/
├── README.md                          # This file
├── requirements/                      # Python dependencies
│   ├── base.txt                       # Core dependencies (PyTorch, Gymnasium, etc.)
│   ├── dev.txt                        # Development tools (pytest, black, ruff, etc.)
│   ├── colab.txt                      # Google Colab minimal dependencies
│   ├── kaggle.txt                     # Kaggle minimal dependencies
│   ├── hierarchical.txt               # MineLand/LLM dependencies (Ollama, etc.)
│   └── all.txt                        # All dependencies combined
├── scripts/                           # Setup and deployment scripts
│   ├── kaggle_setup.py                # Kaggle/Colab environment setup
│   ├── verify_setup.sh                # Repository verification script
│   ├── setup_cluster_image.sh         # Convert Docker to Singularity/Apptainer
│   ├── run_cluster.sh                 # Cluster job submission (Docker)
│   └── run_cluster_sif.sh             # Cluster job submission (Singularity)
└── docker/                            # Docker configuration
    ├── Dockerfile                     # CPU development image
    ├── Dockerfile.gpu                 # GPU cluster image (CUDA 12.1)
    ├── docker-compose.yml             # Development environment orchestration
    ├── .dockerignore                  # Files excluded from Docker build
    └── .env.example                   # Environment variables template
```

## Quick Start

### Local Development (Docker)

From the repository root:

```bash
# Build and start development environment
make build
make up

# Enter the container
make shell

# Stop the environment
make down
```

### Python Dependencies

#### Install Core Dependencies

```bash
pip install -r setup/requirements/base.txt
```

#### Install Development Tools

```bash
pip install -r setup/requirements/dev.txt
```

This includes all base dependencies plus pytest, black, ruff, mypy, jupyter, etc.

#### Install Everything

```bash
pip install -r setup/requirements/all.txt
```

Includes core, development, and hierarchical dependencies.

#### Google Colab / Kaggle

```python
# In a Colab/Kaggle notebook
!pip install -q -r setup/requirements/colab.txt
# or
!pip install -q -r setup/requirements/kaggle.txt
```

## Requirements Files Explained

### base.txt
Core production dependencies:
- PyTorch (CPU-only by default)
- NumPy, SciPy
- Gymnasium, PettingZoo (RL frameworks)
- Matplotlib, ImageIO (visualization)
- TensorBoard, W&B (logging)
- Additional utilities

**Note:** Uses CPU-only PyTorch to avoid large CUDA downloads. GPU will still work via Docker GPU passthrough or when installing with CUDA index.

### dev.txt
Development tools on top of base.txt:
- pytest, pytest-cov (testing)
- black, ruff (formatting, linting)
- mypy (type checking)
- jupyter, ipython (notebooks)
- pre-commit (git hooks)

### colab.txt / kaggle.txt
Minimal dependencies for cloud notebooks. These environments come with PyTorch, NumPy, and other libraries pre-installed, so we only add what's missing.

### hierarchical.txt
Optional dependencies for hierarchical RL experiments:
- ollama (LLM client)
- Pillow (image processing for vision models)
- MineLand (Minecraft RL platform)

**Note:** MineLand requires Java JDK 17, Node.js 18, and additional system dependencies. Use Docker for easiest setup.

### all.txt
Combines base, dev, and hierarchical. Use this if you want the full development environment with all optional features.

## Docker Configuration

### Dockerfile
CPU-optimized development image:
- Python 3.11-slim base
- BuildKit caching for faster rebuilds
- Optional DEV_MODE and HIERARCHICAL_MODE build args
- Non-root user for security
- Vendored dependencies (multigrid, ai_transport) via PYTHONPATH

### Dockerfile.gpu
GPU-optimized image for cluster deployment:
- NVIDIA CUDA 12.1 with cuDNN
- PyTorch with GPU support
- Same structure as Dockerfile but optimized for training

### docker-compose.yml
Development environment orchestration:
- Mounts repository as /workspace volume
- Preserves file permissions (USER_ID/GROUP_ID)
- Port mappings for Jupyter (8888), TensorBoard (6006), debugger (5678)
- Optional Ollama service for LLM inference (hierarchical profile)

## Scripts

### kaggle_setup.py
Python module for setting up EMPO in Kaggle/Colab notebooks:
- Automatically detects environment (Kaggle vs Colab)
- Configures PYTHONPATH for vendored dependencies
- Installs appropriate requirements file
- Provides helper functions for notebook workflows

**Usage:**
```python
import sys
sys.path.insert(0, '/kaggle/working/empo')  # or /content/empo for Colab

from setup.scripts.kaggle_setup import setup_empo
setup_empo(install_deps=True, quiet=True)
```

### verify_setup.sh
Bash script to verify repository structure:
- Checks required files exist
- Validates directory structure
- Tests Python syntax
- Validates docker-compose.yml
- Checks shell script syntax

**Usage:**
```bash
bash setup/scripts/verify_setup.sh
```

### Cluster Scripts

#### setup_cluster_image.sh
Convert Docker image to Singularity/Apptainer SIF format for HPC clusters:
```bash
bash setup/scripts/setup_cluster_image.sh
```

#### run_cluster.sh / run_cluster_sif.sh
Job submission scripts for SLURM/PBS clusters:
```bash
bash setup/scripts/run_cluster.sh           # Docker-based
bash setup/scripts/run_cluster_sif.sh       # Singularity-based
```

## Environment Variables

Create a `.env` file from the template:

```bash
cp setup/docker/.env.example .env
```

Key variables:
- `USER_ID` / `GROUP_ID`: For file permission mapping (auto-detected)
- `WANDB_API_KEY`: W&B authentication
- `HIERARCHICAL_MODE`: Enable MineLand/LLM dependencies (default: false)
- `HOST_JUPYTER_PORT` / `HOST_TENSORBOARD_PORT`: Port mappings

## Building Images

### CPU Image (Local Development)

```bash
make build
```

Builds the CPU image with development dependencies.

### GPU Image (Cluster Deployment)

```bash
make build-gpu
```

Builds the GPU image with CUDA support.

### Hierarchical Image

```bash
make build-hierarchical        # CPU + hierarchical deps
make build-gpu-hierarchical    # GPU + hierarchical deps
```

Builds with MineLand and Ollama support.

## Common Tasks

### Add a New Dependency

1. Edit the appropriate requirements file in `setup/requirements/`
2. Rebuild the Docker image: `make build`
3. Restart containers: `make restart`

### Run Tests

```bash
make test
```

Runs pytest inside the Docker container.

### Lint Code

```bash
make lint
```

Runs ruff and black checks.

### Clean Outputs

```bash
make clean
```

Removes generated outputs, caches, and temporary files.

## Troubleshooting

### Docker build fails with "no such file or directory"
- Ensure you're running commands from the repository root
- Check that `setup/requirements/` files exist

### Permission errors with mounted volumes
- Set USER_ID/GROUP_ID in .env to match your host user
- Restart containers: `make restart`

### Import errors in Python
- Verify PYTHONPATH includes vendored dependencies
- In Docker: automatically configured
- Outside Docker: see main README.md

### Port conflicts
- Set custom ports in .env: `HOST_JUPYTER_PORT=8889`, etc.
- Or use `HOST_JUPYTER_PORT=0` for auto-select

## Related Documentation

- [Main README](../README.md) - Project overview and usage
- [QUICKSTART](../docs/QUICKSTART.md) - Getting started guide
- [CONTRIBUTING](../docs/CONTRIBUTING.md) - Development guidelines
- [PREBUILT_IMAGES](../docs/PREBUILT_IMAGES.md) - Docker Hub images

## Questions?

See the main [README.md](../README.md) or open an issue on GitHub.
