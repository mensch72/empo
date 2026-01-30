# Parameter Sweep Experiments

This directory contains scripts for running parameter sweep experiments on the EMPO framework.

## Overview

The parameter sweep studies how EMPO parameters influence robot behavior in the `asymmetric_freeing_simple.yaml` environment, where a robot must choose which of two locked-in humans to free first.

## Files

- `parameter_sweep_asymmetric_freeing.py`: Main Monte Carlo simulation script
- `analyze_parameter_sweep.py`: Logistic regression analysis script
- `README.md`: This file

## Quick Start

### 1. Run a Small Test (Locally)

```bash
# Test with 10 samples (takes ~5-30 minutes depending on hardware)
python experiments/parameter_sweep_asymmetric_freeing.py \
    --n_samples 10 \
    --output outputs/parameter_sweep/test_results.csv \
    --quiet
```

### 2. Analyze Results

```bash
# Run logistic regression analysis
python experiments/analyze_parameter_sweep.py \
    outputs/parameter_sweep/test_results.csv \
    --interactions \
    --output outputs/parameter_sweep/analysis.txt \
    --plots_dir outputs/parameter_sweep/plots
```

This will:
- Print univariate and multivariate regression results
- Save detailed analysis to `analysis.txt`
- Create visualization plots in `plots/`

### 3. Full Run (HPC)

For production runs with 100+ samples, use the HPC batch script:

```bash
# Copy to your HPC system and submit
sbatch scripts/run_parameter_sweep.sh
```

## Parameters Varied

The experiment varies the following parameters:

| Parameter | Range | Distribution | Description |
|-----------|-------|--------------|-------------|
| `max_steps` | 8-14 | Uniform discrete | Planning horizon |
| `beta_h` | 5-100 | Log-uniform | Human inverse temperature |
| `gamma_h` | 0.8-1.0 | Uniform | Human discount factor |
| `gamma_r` | 0.8-1.0 | Uniform | Robot discount factor |
| `zeta` | 1-3 | Uniform | Risk aversion for goal achievement |
| `eta` | 1-2 | Uniform | Intertemporal power-inequality aversion |
| `xi` | 1-2 | Uniform | Inter-human power-inequality aversion |

**Note:** `beta_r` (robot policy concentration) is held fixed at 50.0.

## Output

### CSV Results File

Each row contains:
- Sampled parameter values
- `left_freed_first`: 1 if left human freed first, 0 if right, -1 if neither
- `left_freed_step`, `right_freed_step`: Steps when each human was freed
- `n_states`: Number of states in computed DAG
- `computation_time`: Total time for backward induction (seconds)

### Analysis Output

- **Text summary** (`analysis.txt`): Regression coefficients, p-values, odds ratios
- **Coefficient plot** (`plots/coefficient_plot.png`): Visual of parameter effects
- **Scatter plots** (`plots/scatter_plots.png`): Relationship between each parameter and P(left)

## Interpreting Results

### Odds Ratios

- **OR > 1**: Increasing this parameter makes it MORE likely the left human is freed first
- **OR < 1**: Increasing this parameter makes it LESS likely the left human is freed first
- **OR ≈ 1**: This parameter has little effect

### Interaction Effects

When using `--interactions`, the analysis includes terms like:
- `beta_h:gamma_h`: Human planning horizon affects how beta_h influences behavior
- `gamma_r:gamma_h`: Discount factor interactions
- `zeta:xi`: Risk aversion parameter interactions

## Computational Requirements

### Local Testing (10 samples)

- **Time**: ~5-30 minutes
- **Memory**: ~2-4 GB
- **CPUs**: 1 (or use `--parallel` with more CPUs)

### Full Run (100 samples)

- **Time**: ~1-5 hours (depends on max_steps and parallelization)
- **Memory**: ~4-8 GB
- **CPUs**: 4-16 recommended with `--parallel`

### HPC Recommendations

For HPC runs with Singularity/Apptainer (no Docker):
```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
```

#### Container Deployment (Recommended for HPC)

Most HPC systems don't support Docker but provide Singularity/Apptainer. The script automatically detects and uses the container runtime.

**Setup (one time):**
```bash
# On your local machine: build and push image to Docker Hub
make up-gpu-docker-hub  # Requires Docker Hub account

# Or build SIF file locally and transfer
make up-gpu-sif-file
scp empo-gpu.sif user@cluster:~/bega/empo/
```

**On HPC cluster:**
```bash
# Pull image from Docker Hub
cd ~/bega/empo
apptainer pull empo.sif docker://your-docker-hub-username/empo:gpu-latest

# Clone repository
mkdir -p git
cd git
git clone https://github.com/yourusername/empo.git .

# Submit job (image auto-detected)
sbatch scripts/run_parameter_sweep.sh
```

The script will automatically:
- Detect `apptainer` or `singularity` command
- Find the SIF file in standard locations
- Bind mount the repository into the container
- Run Python scripts inside the container

**Custom image location:**
```bash
sbatch --export=IMAGE_PATH=/path/to/custom.sif scripts/run_parameter_sweep.sh
```

**Without container (native Python):**
```bash
# If you have Python/dependencies installed natively
sbatch --export=USE_CONTAINER=false scripts/run_parameter_sweep.sh
```

See [CLUSTER.md](../CLUSTER.md) for complete deployment instructions.

#### Native Python (Alternative)

If not using containers, install dependencies first:
```bash
pip install numpy scipy scikit-learn PyYAML cloudpickle tqdm pandas statsmodels matplotlib gymnasium
```

Then use `--parallel --num_workers 8` to leverage multiple cores.

## Example Workflow

```bash
# 1. Small local test
python experiments/parameter_sweep_asymmetric_freeing.py \
    --n_samples 10 \
    --output outputs/parameter_sweep/test.csv

# 2. Analyze test results
python experiments/analyze_parameter_sweep.py \
    outputs/parameter_sweep/test.csv

# 3. If results look good, run full experiment on HPC
sbatch scripts/run_parameter_sweep.sh

# 4. After HPC run completes, download results and analyze
python experiments/analyze_parameter_sweep.py \
    outputs/parameter_sweep/full_results.csv \
    --interactions \
    --output outputs/parameter_sweep/full_analysis.txt
```

## Troubleshooting

### "No humans freed" in many rollouts

- Try increasing `max_steps` range or using higher `beta_r`
- The environment may be too constrained for current parameters

### Very few significant effects

- Increase `n_samples` for more statistical power
- Check if parameters have sufficient variation in your samples
- Some parameters may genuinely have weak effects

### Out of memory errors

- Reduce `max_steps` upper limit (state space grows exponentially)
- Use `--parallel` cautiously (each worker needs memory)
- For HPC, request more memory

## References

- See `docs/API.md` for EMPO parameter definitions
- See main README.md for theoretical background
- Original paper: https://arxiv.org/html/2508.00159v2
