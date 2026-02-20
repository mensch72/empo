# Implementation Summary

## Overview

This repository provides:
1. **EMPO Framework**: Implementation of the two-phase approach for computing AI policies that softly maximize aggregate human power (see [the theoretical paper](https://arxiv.org/html/2508.00159v2))
2. **Unified Docker Environment**: Development and cluster deployment for multi-agent environments

## EMPO Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Human Policy Prior                                         │
│  Compute goal-conditioned Boltzmann policies via backward induction  │
│  → See: docs/API.md, docs/PARALLELIZATION.md                         │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 2: Robot Policy + Power Metric                                │
│  Simultaneously compute robot policy and power metric                │
│  → See: docs/WARMUP_DESIGN.md, docs/BATCHED_COMPUTATION.md           │
├─────────────────────────────────────────────────────────────────────┤
│  World Models: MultiGrid, Transport                                  │
│  → See: docs/API.md, TRANSPORT_ENVIRONMENT.md                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Documentation

| Document | Description |
|----------|-------------|
| [docs/API.md](docs/API.md) | Complete API reference |
| [docs/WARMUP_DESIGN.md](docs/WARMUP_DESIGN.md) | Phase 2 warm-up strategy |
| [docs/ENCODER_ARCHITECTURE.md](docs/ENCODER_ARCHITECTURE.md) | Neural encoder design |
| [docs/BATCHED_COMPUTATION.md](docs/BATCHED_COMPUTATION.md) | Efficient batched computation |
| [docs/EXPLORATION.md](docs/EXPLORATION.md) | Exploration and curiosity methods |
| [docs/PARALLELIZATION.md](docs/PARALLELIZATION.md) | Parallel backward induction |
| [docs/FAQ.md](docs/FAQ.md) | Frequently asked questions |

### Source Code Structure

```
src/empo/                      # Core framework
├── world_model.py             # WorldModel base class
├── possible_goal.py           # Goals, generators, samplers
├── robot_policy.py            # RobotPolicy base class
├── backward_induction/        # Backward induction (tabular)
│   ├── phase1.py              # Phase 1 computation (human policies)
│   └── phase2.py              # Phase 2 computation (robot policies)
├── human_policy_prior.py      # Human policy implementations
├── learning_based/            # Neural network implementations
│   ├── phase1/                # Base classes for Phase 1
│   ├── phase2/                # Phase 2 training (trainer, config, networks)
│   ├── multigrid/             # MultiGrid-specific implementations
│   │   ├── phase1/            # MultiGrid Phase 1 networks
│   │   └── phase2/            # MultiGrid Phase 2 networks
│   └── transport/             # Transport-specific implementations
│       └── phase1/            # Transport Phase 1 networks
└── util/                      # Utilities (memory_monitor, etc.)

vendor/multigrid/              # Extended MultiGrid environment
vendor/ai_transport/           # Transport environment
```

## Docker Infrastructure

### 1. Unified Dockerfile (`Dockerfile`)

- **Base Image**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
  - Full CUDA support for GPU acceleration
  - Compatible with Singularity/Apptainer `--nv` flag
  
- **Dependencies**:
  - Python 3.10
  - PyTorch >= 2.0.0 (with CUDA support)
  - Gymnasium, PettingZoo, Multigrid
  - OpenMPI for distributed training
  - TensorBoard, Weights & Biases for logging
  
- **Build Arguments**:
  - `DEV_MODE`: When set to `true`, installs development dependencies (linters, pytest, etc.)
  - `USER_ID`/`GROUP_ID`: For matching host user permissions
  
- **Security**: Non-root user (`appuser`) for better security

### 2. Docker Compose Setup (`docker-compose.yml`)

- **GPU Support**: Automatic GPU passthrough using Docker Compose GPU syntax
- **Volume Mounting**: Bind-mounts repository to `/workspace` for live development
- **Ports Exposed**:
  - 8888: Jupyter Notebook
  - 6006: TensorBoard
  - 5678: Python debugger (debugpy)
- **Environment Variables**: Supports `CUDA_VISIBLE_DEVICES`, `WANDB_API_KEY`
- **Persistent Cache**: Optional pip cache volume

### 3. Training Script (`train.py`)

- Complete CLI interface with argparse
- GPU detection and usage
- Configurable hyperparameters
- Output directory management
- Placeholder for MARL training logic
- Easy to extend for actual implementation

### 4. Project Structure

```
empo/
├── Dockerfile                    # Unified image definition
├── docker-compose.yml            # Local development
├── setup/                             # Setup and packaging helpers
│   ├── requirements.txt              # Core dependencies
│   └── requirements-dev.txt          # Dev dependencies
├── src/empo/                     # Core EMPO framework
├── vendor/multigrid/             # MultiGrid environment
├── vendor/ai_transport/          # Transport environment
├── examples/                     # Usage examples
├── tests/                        # Test infrastructure
├── docs/                         # Detailed documentation
├── setup/scripts/
│   ├── run_cluster.sh           # SLURM job script
│   ├── setup_cluster_image.sh   # Cluster setup helper
│   └── verify_setup.sh          # Verification script
├── Makefile                      # Convenience commands
├── README.md                     # Full documentation
├── QUICKSTART.md                 # Quick start guide
└── CLUSTER.md                    # Cluster deployment guide
```

### 5. Cluster Deployment Support

**Singularity/Apptainer Ready**:
- Same Dockerfile works for both Docker and Singularity
- CUDA runtime compatible with `--nv` flag
- Example SLURM job script (`setup/scripts/run_cluster.sh`)
- Setup helper script (`setup/scripts/setup_cluster_image.sh`)
- Bind mount support for repository access

**Workflow**:
1. Build Docker image locally or in CI
2. Push to Docker registry (DockerHub, GHCR, etc.)
3. Pull on cluster: `apptainer pull empo.sif docker://user/empo:latest`
4. Run: `apptainer exec --nv -B $(pwd):/workspace empo.sif python /workspace/train.py`
5. Or submit SLURM job: `sbatch setup/scripts/run_cluster.sh`

### 6. Development Tools

**Makefile**: Convenient commands for common tasks
- `make up/down`: Start/stop environment
- `make shell`: Enter container
- `make train`: Run training
- `make clean`: Clean outputs

**Verification**: `setup/scripts/verify_setup.sh` checks all components

**Testing**: Basic test infrastructure in `tests/`

### 7. Documentation

- **README.md**: Comprehensive guide covering:
  - Local development workflow
  - Cluster deployment workflow
  - GPU support setup
  - Troubleshooting
  - Advanced usage
  
- **QUICKSTART.md**: Get running in 5 minutes
  - Quick setup steps
  - Common commands
  - Quick troubleshooting

## Design Decisions

### Single Dockerfile Approach

**Why**: Avoids duplicate configurations and maintenance burden

**How**: 
- Base image includes production dependencies
- `DEV_MODE` build arg conditionally adds dev tools
- Docker Compose sets `DEV_MODE=true`
- Cluster builds use default `DEV_MODE=false`

### GPU Support Strategy

**Docker Compose**: Uses native GPU device reservation syntax
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Singularity/Apptainer**: Uses `--nv` flag
```bash
apptainer exec --nv image.sif python script.py
```

### Bind Mount Pattern

**Local Dev**: Repository is bind-mounted for live editing
```yaml
volumes:
  - .:/workspace
```

**Cluster**: Same pattern with `-B` flag
```bash
-B /path/to/repo:/workspace
```

## Usage Examples

### Local Development

```bash
# Start
docker compose up -d

# Enter
docker compose exec empo-dev bash

# Train
python train.py --num-episodes 1000

# With GPU
python train.py --env-name CartPole-v1 --num-episodes 1000
```

### Cluster Deployment

```bash
# Build and push
docker build -t user/empo:latest .
docker push user/empo:latest

# On cluster
apptainer pull empo.sif docker://user/empo:latest

# Run
apptainer exec --nv -B $(pwd):/workspace empo.sif \
  python /workspace/train.py --num-episodes 10000

# Submit job
sbatch setup/scripts/run_cluster.sh
```

## Testing & Validation

### Structure Tests
```bash
python tests/test_structure.py
```

### Full Verification
```bash
bash setup/scripts/verify_setup.sh
```

### Docker Build Test
```bash
docker build -t empo:test .
```

### Compose Test
```bash
docker compose config --quiet
```

## Extension Points

### Adding Dependencies

Edit `setup/requirements.txt` or `setup/requirements-dev.txt`, then rebuild:
```bash
docker compose up --build
```

### Adding Environments

Install in requirements:
```txt
multigrid>=0.1.0
your-custom-env>=1.0.0
```

## Known Limitations

1. **Network Requirements**: 
   - Initial build requires network access for apt and pip
   - Cluster nodes typically don't have network during jobs
   - Solution: Pre-build and distribute image

2. **Size**: 
   - CUDA base images are large (~4-5GB)
   - Consider using smaller base for CPU-only deployments

3. **MPI Support**: 
   - Basic MPI support included
   - Distributed training may need additional setup

## Security Considerations

- Non-root user in container
- No hardcoded credentials
- `.env` file for sensitive variables (gitignored)
- Secrets should be passed via environment or mounted files

## Maintenance

To update:
1. Dependencies: Edit requirements files
2. CUDA version: Update Dockerfile base image
3. Python version: Update Dockerfile apt-get
4. Documentation: Update README.md and QUICKSTART.md

## Future Enhancements

Potential additions:
- Pre-commit hooks for code quality
- CI/CD pipeline for automated builds
- Multi-stage Dockerfile for smaller images
- Additional example environments
- Advanced distributed training support
- Jupyter Lab integration
- VS Code Remote Container config

## Related Documentation

- [README.md](README.md) - Project overview and theory explanation
- [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
- [CLUSTER.md](CLUSTER.md) - HPC cluster deployment
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [docs/PREBUILT_IMAGES.md](docs/PREBUILT_IMAGES.md) - Pre-built container images
