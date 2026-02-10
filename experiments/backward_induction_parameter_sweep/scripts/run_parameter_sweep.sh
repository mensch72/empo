#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=empo_param_sweep
#SBATCH --account=bega
#SBATCH --output=outputs/parameter_sweep/logs/slurm_%j.out
#SBATCH --error=outputs/parameter_sweep/logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# Parameter Sweep HPC Batch Script (Parallel Tasks)
# 
# This script runs multiple parallel tasks, each computing n_samples new samples.
# All tasks append to the same results.csv file (with file locking).
#
# Total samples = ntasks * n_samples
#
# Usage:
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh [OPTIONS]
#
# Options:
#   -n, --n_samples N       Number of NEW samples PER TASK (default: 10)
#   -o, --output FILE       Output CSV file (default: outputs/parameter_sweep/results.csv)
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
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh
#
#   # Quick test run
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh --quick
#
#   # 100 samples per task (with 4 tasks = 400 total)
#   sbatch experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh -n 100
#
#   # Run 10 tasks with 50 samples each (500 total)
#   sbatch --ntasks=10 experiments/backward_induction_parameter_sweep/scripts/run_parameter_sweep.sh -n 50

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
N_SAMPLES=10
OUTPUT_FILE="outputs/parameter_sweep/results.csv"
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

# Create output directories
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "outputs/parameter_sweep/logs"

# Print job info
echo "=================================================="
echo "EMPO Parameter Sweep (Task $SLURM_PROCID of $SLURM_NTASKS)"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Samples per task: $N_SAMPLES"
echo "Total tasks: ${SLURM_NTASKS:-1}"
echo "Output file: $OUTPUT_FILE"
echo "Quick mode: $QUICK_MODE"
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
echo "Started: $(date)"
echo "=================================================="
echo ""

# Build command arguments
PYTHON_ARGS=(
    "experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py"
    "--n_samples" "$N_SAMPLES"
    "--output" "$OUTPUT_FILE"
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

# Add quick mode flag
if [ "$QUICK_MODE" = "true" ]; then
    PYTHON_ARGS+=("--quick")
fi

# Run the parameter sweep
# Use srun to launch parallel tasks under SLURM, or direct python locally
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Starting ${SLURM_NTASKS:-1} parallel tasks via srun..."
    # Each task gets its own output file via srun's --output/--error options
    srun --output="outputs/parameter_sweep/logs/slurm_${SLURM_JOB_ID}_%t.out" \
         --error="outputs/parameter_sweep/logs/slurm_${SLURM_JOB_ID}_%t.err" \
         python -u "${PYTHON_ARGS[@]}"
else
    echo "Starting parameter sweep (local mode)..."
    python -u "${PYTHON_ARGS[@]}"
fi

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "All tasks completed successfully!"
else
    echo "ERROR: Some tasks failed (exit code $EXIT_CODE)"
fi
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
