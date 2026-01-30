# HPC Deployment Guide for Parameter Sweep

This guide explains how to run the parameter sweep experiment on HPC clusters using Singularity/Apptainer (no Docker required).

## Quick Start

### 1. Build/Pull Container Image (One Time)

Choose one of these methods:

**Method A: Pull from Docker Hub (Easiest)**
```bash
# On your local machine: build and push
docker login
make up-gpu-docker-hub  # Requires .env with DOCKER_USERNAME

# On HPC cluster: pull
cd ~/bega/empo
apptainer pull empo.sif docker://your-docker-hub-username/empo:gpu-latest
```

**Method B: Build SIF Locally and Transfer**
```bash
# On local machine (requires Apptainer/Singularity)
make up-gpu-sif-file
scp empo-gpu.sif user@cluster:~/bega/empo/

# On HPC cluster: rename if needed
cd ~/bega/empo
mv empo-gpu.sif empo.sif
```

**Method C: Build Directly on HPC**
```bash
# On HPC cluster (if Docker daemon available)
cd ~/bega/empo/git
apptainer build ../empo.sif Dockerfile
```

### 2. Clone Repository

```bash
cd ~/bega/empo
mkdir -p git
cd git
git clone https://github.com/yourusername/empo.git .
```

### 3. Submit Job

```bash
cd ~/bega/empo/git
sbatch scripts/run_parameter_sweep.sh
```

That's it! The script automatically:
- Detects `apptainer` or `singularity`
- Finds the SIF file in standard locations
- Runs everything inside the container

## Customization

### Change Number of Samples

```bash
sbatch --export=N_SAMPLES=200 scripts/run_parameter_sweep.sh
```

### Specify Custom Image Path

```bash
sbatch --export=IMAGE_PATH=/path/to/custom.sif scripts/run_parameter_sweep.sh
```

### Use Native Python (No Container)

If you have Python and dependencies installed:

```bash
# Install dependencies first
pip install --user numpy scipy scikit-learn PyYAML cloudpickle tqdm pandas statsmodels matplotlib gymnasium

# Submit job
sbatch --export=USE_CONTAINER=false scripts/run_parameter_sweep.sh
```

### Multiple Parameters

```bash
sbatch --export=N_SAMPLES=200,N_ROLLOUTS=10,SEED=123 scripts/run_parameter_sweep.sh
```

## Directory Structure

The script expects this layout:

```
~/bega/empo/
├── empo.sif                    # Container image (auto-detected here)
├── empo-gpu.sif               # (or this name)
└── git/                       # Repository clone
    ├── experiments/
    │   ├── parameter_sweep_asymmetric_freeing.py
    │   └── analyze_parameter_sweep.py
    ├── scripts/
    │   └── run_parameter_sweep.sh
    └── outputs/
        └── parameter_sweep/   # Results go here
            ├── results_*.csv
            ├── analysis_*.txt
            └── plots_*/
```

## Troubleshooting

### "Neither apptainer nor singularity found"

**Solution**: Your HPC doesn't have Apptainer/Singularity installed. Either:
1. Ask your HPC admin to install it (standard on most clusters)
2. Use native Python: `sbatch --export=USE_CONTAINER=false ...`

### "Could not auto-detect image path"

**Solution**: Place the SIF file in a standard location:
```bash
cd ~/bega/empo
# Move/copy your image here
cp /path/to/empo.sif .
```

Or specify the path explicitly:
```bash
sbatch --export=IMAGE_PATH=/full/path/to/empo.sif scripts/run_parameter_sweep.sh
```

### "Image not found at ..."

**Solution**: The image doesn't exist at the detected path. Check:
```bash
ls -lh ~/bega/empo/*.sif
```

If missing, pull or build the image (see step 1 above).

### Job fails with Python import errors

**Possible causes**:
1. Container image is outdated - rebuild/pull latest
2. Using native Python but dependencies not installed
3. Wrong Python version

**Solutions**:
```bash
# If using container: rebuild/pull latest image
cd ~/bega/empo
apptainer pull --force empo.sif docker://youruser/empo:gpu-latest

# If using native Python: install dependencies
pip install --user numpy scipy scikit-learn PyYAML cloudpickle tqdm pandas statsmodels matplotlib gymnasium
```

### "Out of memory" errors

**Solution**: Request more memory in SBATCH directives. Edit `scripts/run_parameter_sweep.sh`:
```bash
#SBATCH --mem=32G  # Increase from 16G
```

Or reduce max_steps range to decrease state space size. Edit `experiments/parameter_sweep_asymmetric_freeing.py`:
```python
max_steps = np.random.randint(8, 11)  # Instead of 8, 15
```

### Very slow execution

**Possible causes**:
1. max_steps too large (exponential state space growth)
2. Not using parallelization

**Solutions**:
```bash
# Increase CPUs and use parallel mode
#SBATCH --cpus-per-task=16  # Edit script

# Script already uses: --parallel --num_workers $SLURM_CPUS_PER_TASK
```

## Resource Recommendations

### Small Test (10 samples)
```bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
```

### Medium Run (50 samples)
```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
```

### Full Run (100+ samples)
```bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
```

## Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Output (Live)
```bash
tail -f outputs/parameter_sweep/slurm_*.out
```

### View Errors
```bash
tail -f outputs/parameter_sweep/slurm_*.err
```

### Cancel Job
```bash
scancel <job_id>
```

## After Job Completes

Results are in `outputs/parameter_sweep/`:
- `results_<timestamp>_n<samples>.csv` - Raw data
- `analysis_<timestamp>.txt` - Regression results
- `plots_<timestamp>/` - Visualizations

Download to local machine:
```bash
scp -r user@cluster:~/bega/empo/git/outputs/parameter_sweep/ ./results/
```

## See Also

- [CLUSTER.md](../CLUSTER.md) - Complete cluster deployment guide
- [experiments/README.md](README.md) - Parameter sweep overview
- [experiments/USAGE_EXAMPLE.md](USAGE_EXAMPLE.md) - Usage examples
