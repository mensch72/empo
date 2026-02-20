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

Two scripts are available:

**Native Python (if dependencies installed on cluster):**
```bash
cd ~/bega/empo/git
sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh
```

**With Apptainer/Singularity container:**
```bash
cd ~/bega/empo/git
sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep_apptainer.sh
```

The apptainer script automatically:
- Detects `apptainer` or `singularity`
- Finds the SIF file in standard locations
- Runs everything inside the container

## Customization

Both scripts accept command-line arguments for all parameters.

### Change Number of Samples

```bash
sbatch .../run_parameter_sweep.sh -n 200
```

### Quick Test Run

```bash
sbatch .../run_parameter_sweep.sh --quick
```

### Specify Custom Image Path (apptainer script only)

```bash
sbatch .../run_parameter_sweep_apptainer.sh --image /path/to/custom.sif
```

### Custom Prior Bounds

```bash
sbatch .../run_parameter_sweep.sh \
    --beta_h_min 10 --beta_h_max 50 \
    --gamma_h_min 0.9 --gamma_h_max 1.0
```

### Multiple Parameters

```bash
sbatch .../run_parameter_sweep.sh \
    -n 200 -s 123 --max_steps_min 10 --max_steps_max 12
```

## Directory Structure

The script expects this layout:

```
~/bega/empo/
├── empo.sif                    # Container image (auto-detected here)
├── empo-gpu.sif               # (or this name)
└── git/                       # Repository clone
    ├── experiments/
    │   └── backward_induction_parameter_sweep/
    │       ├── parameter_sweep_asymmetric_freeing.py
    │       ├── analyze_parameter_sweep.py
    │       └── scripts/
    │           ├── run_parameter_sweep.sh          # Native Python
    │           └── run_parameter_sweep_apptainer.sh # With container
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
sbatch .../run_parameter_sweep_apptainer.sh --image /full/path/to/empo.sif
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

**Solution**: Request more memory in SBATCH directives. Edit the script:
```bash
#SBATCH --mem=32G  # Increase from 16G
```

Or reduce max_steps range via command line:
```bash
sbatch .../run_parameter_sweep.sh --max_steps_min 8 --max_steps_max 11
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

- [CLUSTER.md](../../CLUSTER.md) - Complete cluster deployment guide
- [README.md](README.md) - Parameter sweep overview
- [USAGE_EXAMPLE.md](USAGE_EXAMPLE.md) - Usage examples
