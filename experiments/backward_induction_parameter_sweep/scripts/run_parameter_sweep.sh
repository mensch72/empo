#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=empo_param_sweep
#SBATCH --account=bega
#SBATCH --output=outputs/parameter_sweep/slurm_%j.out
#SBATCH --error=outputs/parameter_sweep/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=1G

# Parameter Sweep HPC Batch Script (Native Python)
# 
# This script runs the parameter sweep experiment using the host's Python
# environment. Edit the "Environment Setup" section below to activate your
# Python environment.
#
# Adjust SBATCH directives above according to your cluster's configuration.
#
# Usage:
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh [OPTIONS]
#
# Options:
#   -n, --n_samples N       Number of samples (default: 100)
#   -s, --seed SEED         Random seed (default: 42)
#   -o, --output FILE       Output CSV file (auto-generated if not specified)
#   --quick                 Quick mode (small max_steps, few samples)
#
# Prior bound options:
#   --max_steps_min N       Minimum max_steps (default: 8)
#   --max_steps_max N       Maximum max_steps (default: 14)
#   --beta_h_min F          Minimum beta_h (default: 5.0)
#   --beta_h_max F          Maximum beta_h (default: 100.0)
#   --gamma_h_min F         Minimum gamma_h (default: 0.8)
#   --gamma_h_max F         Maximum gamma_h (default: 1.0)
#   --gamma_r_min F         Minimum gamma_r (default: 0.8)
#   --gamma_r_max F         Maximum gamma_r (default: 1.0)
#   --zeta_min F            Minimum zeta (default: 1.0)
#   --zeta_max F            Maximum zeta (default: 3.0)
#   --eta_min F             Minimum eta (default: 1.0)
#   --eta_max F             Maximum eta (default: 2.0)
#   --xi_min F              Minimum xi (default: 1.0)
#   --xi_max F              Maximum xi (default: 2.0)
#
# Examples:
#   # Basic run with defaults
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh
#
#   # Quick test run
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh --quick
#
#   # Custom samples and seed
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh -n 200 -s 123
#
#   # Custom prior bounds
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh --beta_h_min 10 --beta_h_max 50

set -e  # Exit on error

#==============================================================================
# Environment Setup - EDIT THIS SECTION
#==============================================================================
# Uncomment and modify the lines below to activate your Python environment:
#
# Option 1: Conda environment
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate empo
#
# Option 2: Virtual environment
# source /path/to/venv/bin/activate
#
# Option 3: Module system
# module load python/3.10
# module load anaconda3
module load anaconda/2025
source activate empo
export PATH="$CONDA_PREFIX/bin:$PATH"
#==============================================================================

# Default configuration
N_SAMPLES=100
SEED=42
OUTPUT_FILE=""
QUICK_MODE=false

# Prior bound defaults (matching Python script)
MAX_STEPS_MIN=8
MAX_STEPS_MAX=8
BETA_H_MIN=5.0
BETA_H_MAX=100.0
GAMMA_H_MIN=0.8
GAMMA_H_MAX=1.0
GAMMA_R_MIN=0.8
GAMMA_R_MAX=1.0
ZETA_MIN=1.0
ZETA_MAX=3.0
ETA_MIN=1.0
ETA_MAX=2.0
XI_MIN=1.0
XI_MAX=2.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --max_steps_min)
            MAX_STEPS_MIN="$2"
            shift 2
            ;;
        --max_steps_max)
            MAX_STEPS_MAX="$2"
            shift 2
            ;;
        --beta_h_min)
            BETA_H_MIN="$2"
            shift 2
            ;;
        --beta_h_max)
            BETA_H_MAX="$2"
            shift 2
            ;;
        --gamma_h_min)
            GAMMA_H_MIN="$2"
            shift 2
            ;;
        --gamma_h_max)
            GAMMA_H_MAX="$2"
            shift 2
            ;;
        --gamma_r_min)
            GAMMA_R_MIN="$2"
            shift 2
            ;;
        --gamma_r_max)
            GAMMA_R_MAX="$2"
            shift 2
            ;;
        --zeta_min)
            ZETA_MIN="$2"
            shift 2
            ;;
        --zeta_max)
            ZETA_MAX="$2"
            shift 2
            ;;
        --eta_min)
            ETA_MIN="$2"
            shift 2
            ;;
        --eta_max)
            ETA_MAX="$2"
            shift 2
            ;;
        --xi_min)
            XI_MIN="$2"
            shift 2
            ;;
        --xi_max)
            XI_MAX="$2"
            shift 2
            ;;
        -h|--help)
            head -n 55 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Setup output directory and file
OUTPUT_DIR="outputs/parameter_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/results.csv"
fi

# Print job info
echo "=================================================="
echo "EMPO Parameter Sweep Experiment (Native Python)"
echo "=================================================="
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURM_NODELIST"
    echo "CPUs: $SLURM_CPUS_PER_TASK"
    echo "Memory: $SLURM_MEM_PER_NODE MB"
fi
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  N_SAMPLES: $N_SAMPLES"
echo "  SEED: $SEED"
echo "  OUTPUT_FILE: $OUTPUT_FILE"
echo "  QUICK_MODE: $QUICK_MODE"
echo ""
echo "Prior Bounds:"
echo "  max_steps: [$MAX_STEPS_MIN, $MAX_STEPS_MAX]"
echo "  beta_h:    [$BETA_H_MIN, $BETA_H_MAX]"
echo "  gamma_h:   [$GAMMA_H_MIN, $GAMMA_H_MAX]"
echo "  gamma_r:   [$GAMMA_R_MIN, $GAMMA_R_MAX]"
echo "  zeta:      [$ZETA_MIN, $ZETA_MAX]"
echo "  eta:       [$ETA_MIN, $ETA_MAX]"
echo "  xi:        [$XI_MIN, $XI_MAX]"
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "=================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command arguments
PYTHON_ARGS=(
    "experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py"
    "--n_samples" "$N_SAMPLES"
    "--output" "$OUTPUT_FILE"
    "--parallel"
    "--seed" "$SEED"
    "--max_steps_min" "$MAX_STEPS_MIN"
    "--max_steps_max" "$MAX_STEPS_MAX"
    "--beta_h_min" "$BETA_H_MIN"
    "--beta_h_max" "$BETA_H_MAX"
    "--gamma_h_min" "$GAMMA_H_MIN"
    "--gamma_h_max" "$GAMMA_H_MAX"
    "--gamma_r_min" "$GAMMA_R_MIN"
    "--gamma_r_max" "$GAMMA_R_MAX"
    "--zeta_min" "$ZETA_MIN"
    "--zeta_max" "$ZETA_MAX"
    "--eta_min" "$ETA_MIN"
    "--eta_max" "$ETA_MAX"
    "--xi_min" "$XI_MIN"
    "--xi_max" "$XI_MAX"
)

# Add num_workers if running in SLURM
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    PYTHON_ARGS+=("--num_workers" "$SLURM_CPUS_PER_TASK")
fi

# Add quick mode flag
if [ "$QUICK_MODE" = "true" ]; then
    PYTHON_ARGS+=("--quick")
fi

# Run the parameter sweep (use -u for unbuffered output)
echo "Starting parameter sweep..."
echo "Command: python -u ${PYTHON_ARGS[*]}"
python -u "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Parameter sweep completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "=================================================="
    echo ""
    
    # Run analysis
    echo "Running GLM analysis..."
    python -u experiments/backward_induction_parameter_sweep/analyze_parameter_sweep.py \
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
