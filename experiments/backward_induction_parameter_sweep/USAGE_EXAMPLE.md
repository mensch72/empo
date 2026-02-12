# Usage Example: Parameter Sweep Experiment

This document provides a step-by-step example of running the parameter sweep experiment.

## Quick Start

### 1. Test Run (Local, ~10 minutes)

```bash
# Run a small test with 10 parameter combinations
python experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py \
    --n_samples 10 \
    --n_rollouts 5 \
    --output outputs/parameter_sweep/test_results.csv \
    --seed 42

# Analyze the results
python experiments/backward_induction_parameter_sweep/analyze_parameter_sweep.py \
    outputs/parameter_sweep/test_results.csv \
    --output outputs/parameter_sweep/test_analysis.txt

# View results
cat outputs/parameter_sweep/test_analysis.txt
```

### 2. Production Run (HPC, ~2-6 hours)

For a full scientific study with statistical power, deploy on HPC using Singularity/Apptainer.

#### Option A: Using Container (Recommended - No Docker Needed)

Most HPC clusters provide Singularity/Apptainer instead of Docker. The script automatically uses it.

**One-time setup:**
```bash
# On local machine: Build image and push to Docker Hub
docker login
make up-gpu-docker-hub  # Or use make up-gpu-sif-file

# On HPC: Pull the image
cd ~/bega/empo
apptainer pull empo.sif docker://your-docker-hub-username/empo:gpu-latest

# Clone repository
mkdir -p git && cd git
git clone https://github.com/yourusername/empo.git .
```

**Submit job:**
```bash
cd ~/bega/empo/git
sbatch scripts/run_parameter_sweep.sh

# Or with custom settings
sbatch --export=N_SAMPLES=200,IMAGE_PATH=../empo.sif scripts/run_parameter_sweep.sh
```

The script will automatically:
- Detect if `apptainer` or `singularity` is available
- Find the SIF file (checks standard locations)
- Run everything inside the container with proper bind mounts

#### Option B: Native Python (Without Container)

If your HPC has Python installed natively:

```bash
# Install dependencies (one time)
pip install --user numpy scipy scikit-learn PyYAML cloudpickle tqdm pandas statsmodels matplotlib gymnasium

# Submit job without container
sbatch --export=USE_CONTAINER=false scripts/run_parameter_sweep.sh
```

#### Option C: Manual Run

For full control:

```bash
# On HPC cluster with SLURM
python experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py \
    --n_samples 100 \
    --n_rollouts 10 \
    --output outputs/parameter_sweep/full_results.csv \
    --parallel \
    --num_workers 8 \
    --seed 42
```

## Example Output

### CSV Results

```csv
max_steps,beta_h,beta_r,gamma_h,gamma_r,zeta,eta,xi,left_freed_first,left_freed_step,right_freed_step,n_states,computation_time
8,12.34,50.0,0.89,0.92,1.56,1.23,1.45,1,3,5,1024,45.2
9,67.89,50.0,0.95,0.88,2.34,1.67,1.12,0,5,3,2048,89.7
...
```

### Regression Analysis

```
================================================================================
LOGISTIC REGRESSION RESULTS
================================================================================

Model Fit Statistics:
  Observations: 95
  Log-Likelihood: -58.23
  AIC: 130.46
  BIC: 148.92
  Pseudo R-squared (McFadden): 0.1234

Coefficients:

Variable             Coef    Std Err        z   P>|z|  Odds Ratio   Sig
--------------------------------------------------------------------------------
Intercept           2.3456     1.2345    1.900   0.0574      10.4379      
max_steps          -0.1234     0.0567   -2.177   0.0295       0.8839     *
beta_h              0.0089     0.0045    1.978   0.0479       1.0089     *
gamma_h             1.2345     0.5678    2.174   0.0297       3.4365     *
gamma_r            -0.5678     0.3456   -1.643   0.1004       0.5669      
zeta                0.3456     0.1789    1.932   0.0534       1.4126      
eta                -0.2345     0.2134   -1.099   0.2718       0.7911      
xi                  0.1234     0.1567    0.787   0.4311       1.1313      

Significance: *** p<0.001, ** p<0.01, * p<0.05

================================================================================
INTERPRETATION
================================================================================

Odds Ratio > 1: Increasing this parameter makes it MORE likely to free left human first
Odds Ratio < 1: Increasing this parameter makes it LESS likely to free left human first
Odds Ratio ≈ 1: This parameter has little effect on the decision

Significant Effects (p < 0.05):
  max_steps: decreases P(left) by ~11.6% per unit increase
  beta_h: increases P(left) by ~0.9% per unit increase
  gamma_h: increases P(left) by ~243.7% per unit increase
```

## Interpreting Results

### Key Questions

1. **Which parameters matter most?**
   - Look for significant effects (p < 0.05) with large odds ratios
   - In the example above: `gamma_h` has the strongest effect

2. **What's the direction of effects?**
   - Odds Ratio > 1: Parameter increases P(left)
   - Odds Ratio < 1: Parameter decreases P(left)

3. **Are there interaction effects?**
   - Run with `--interactions` flag to test
   - Example: `beta_h:gamma_h` tests if human planning horizon affects beta_h's influence

### Visualization

The analysis script generates plots in `outputs/parameter_sweep/plots/`:

- `coefficient_plot.png`: Shows effect sizes and confidence intervals
- `scatter_plots.png`: Relationship between each parameter and P(left)

## Customization

### Different Parameter Ranges

Edit `sample_parameters()` in `experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py`:

```python
def sample_parameters(seed: Optional[int] = None) -> ParameterSet:
    # Example: narrower beta_h range
    beta_h = np.exp(np.random.uniform(np.log(10), np.log(50)))  # 10 to 50 instead of 5 to 100
    
    # Example: fixed gamma values
    gamma_h = 0.95  # Fixed instead of random
    gamma_r = 0.95
    
    # ...
```

### Different World

Use `--world` flag to test other environments:

```bash
python experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py \
    --world multigrid_worlds/jobst_challenges/asymmetric_freeing.yaml \
    --n_samples 50
```

### Custom Analysis

Modify `experiments/backward_induction_parameter_sweep/analyze_parameter_sweep.py` to add:
- Different regression models (probit, complementary log-log)
- Non-linear effects (polynomial terms, splines)
- Stratified analysis (subset by max_steps ranges)

## Troubleshooting

### Issue: Very few humans freed

**Solution**: Increase `max_steps` range or adjust parameter priors
```python
max_steps = np.random.randint(10, 16)  # Longer horizon
```

### Issue: Out of memory

**Solution**: Reduce `max_steps` upper limit or use HPC with more RAM
```python
max_steps = np.random.randint(8, 11)  # Smaller state space
```

### Issue: No significant effects found

**Solution**: 
1. Increase `--n_samples` for more statistical power
2. Check if parameters have sufficient variation
3. Try interaction effects with `--interactions`

## Next Steps

After running the experiment:

1. **Publish results**: Share CSV and analysis with team
2. **Iterate**: Based on findings, adjust parameter ranges for focused follow-up
3. **Extend**: Try different environments or add new parameters
4. **Visualize**: Create custom plots for presentations/papers

## Citation

If you use this experiment in published work, cite:

```bibtex
@misc{empo_parameter_sweep,
  title={Parameter Influence Study on EMPO Robot Decision-Making},
  author={[Your Name]},
  year={2026},
  note={Experiment on asymmetric_freeing_simple.yaml environment}
}
```
