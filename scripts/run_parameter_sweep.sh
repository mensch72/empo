#!/bin/bash
#SBATCH --job-name=empo_param_sweep
#SBATCH --output=outputs/parameter_sweep/slurm_%j.out
#SBATCH --error=outputs/parameter_sweep/slurm_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=compute

# Parameter Sweep HPC Batch Script
# 
# This script runs the parameter sweep experiment on an HPC cluster.
# Adjust SBATCH directives above according to your cluster's configuration.
#
# Usage:
#   sbatch scripts/run_parameter_sweep.sh
#
# To customize the number of samples:
#   sbatch --export=N_SAMPLES=200 scripts/run_parameter_sweep.sh

# Configuration
N_SAMPLES=${N_SAMPLES:-100}  # Default: 100 samples
N_ROLLOUTS=${N_ROLLOUTS:-5}   # Default: 5 rollouts per sample
SEED=${SEED:-42}              # Random seed
OUTPUT_DIR="outputs/parameter_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}_n${N_SAMPLES}.csv"

# Print job info
echo "=================================================="
echo "EMPO Parameter Sweep Experiment"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  N_SAMPLES: $N_SAMPLES"
echo "  N_ROLLOUTS: $N_ROLLOUTS"
echo "  SEED: $SEED"
echo "  OUTPUT_FILE: $OUTPUT_FILE"
echo "=================================================="
echo ""

# Load modules (adjust for your HPC system)
# module load python/3.9
# module load anaconda3

# Activate conda environment if needed
# source activate empo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the parameter sweep
echo "Starting parameter sweep..."
python experiments/parameter_sweep_asymmetric_freeing.py \
    --n_samples "$N_SAMPLES" \
    --n_rollouts "$N_ROLLOUTS" \
    --output "$OUTPUT_FILE" \
    --parallel \
    --num_workers "$SLURM_CPUS_PER_TASK" \
    --seed "$SEED" \
    --quiet

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Parameter sweep completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "=================================================="
    echo ""
    
    # Run analysis
    echo "Running logistic regression analysis..."
    python experiments/analyze_parameter_sweep.py \
        "$OUTPUT_FILE" \
        --interactions \
        --output "${OUTPUT_DIR}/analysis_${TIMESTAMP}.txt" \
        --plots_dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"
    
    echo ""
    echo "Analysis complete!"
    echo "  Text results: ${OUTPUT_DIR}/analysis_${TIMESTAMP}.txt"
    echo "  Plots: ${OUTPUT_DIR}/plots_${TIMESTAMP}/"
else
    echo ""
    echo "=================================================="
    echo "ERROR: Parameter sweep failed with exit code $EXIT_CODE"
    echo "Check the log files for details."
    echo "=================================================="
fi

echo ""
echo "Job finished: $(date)"

exit $EXIT_CODE
