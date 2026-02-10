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
# This script runs the parameter sweep experiment on an HPC cluster using
# Singularity/Apptainer containers (no Docker required).
#
# Adjust SBATCH directives above according to your cluster's configuration.
#
# Usage:
#   sbatch scripts/run_parameter_sweep.sh
#
# To customize the number of samples:
#   sbatch --export=N_SAMPLES=200 scripts/run_parameter_sweep.sh
#
# To customize container image path:
#   sbatch --export=IMAGE_PATH=/path/to/empo.sif scripts/run_parameter_sweep.sh
#
# For native Python (without container):
#   sbatch --export=USE_CONTAINER=false scripts/run_parameter_sweep.sh

# Configuration
N_SAMPLES=${N_SAMPLES:-100}  # Default: 100 samples
N_ROLLOUTS=${N_ROLLOUTS:-5}   # Default: 5 rollouts per sample
SEED=${SEED:-42}              # Random seed
OUTPUT_DIR="outputs/parameter_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}_n${N_SAMPLES}.csv"

# Container configuration
USE_CONTAINER=${USE_CONTAINER:-true}  # Set to 'false' to use native Python
REPO_PATH="${REPO_PATH:-$(pwd)}"      # Repository path (auto-detected)
IMAGE_PATH="${IMAGE_PATH:-}"          # Auto-detect if not specified

# Detect container runtime (apptainer or singularity)
if [ "$USE_CONTAINER" = "true" ]; then
    if command -v apptainer &> /dev/null; then
        CONTAINER_CMD="apptainer"
    elif command -v singularity &> /dev/null; then
        CONTAINER_CMD="singularity"
    else
        echo "ERROR: USE_CONTAINER=true but neither apptainer nor singularity found!"
        echo "Please install Apptainer/Singularity or set USE_CONTAINER=false"
        exit 1
    fi
    
    # Auto-detect image path if not specified
    if [ -z "$IMAGE_PATH" ]; then
        # Try common locations
        if [ -f "$(dirname "$REPO_PATH")/empo.sif" ]; then
            IMAGE_PATH="$(dirname "$REPO_PATH")/empo.sif"
        elif [ -f "$(dirname "$REPO_PATH")/empo-gpu.sif" ]; then
            IMAGE_PATH="$(dirname "$REPO_PATH")/empo-gpu.sif"
        elif [ -f "$REPO_PATH/empo.sif" ]; then
            IMAGE_PATH="$REPO_PATH/empo.sif"
        else
            echo "ERROR: Could not auto-detect image path."
            echo "Please specify IMAGE_PATH or place empo.sif in a standard location."
            echo "Standard locations:"
            echo "  $(dirname "$REPO_PATH")/empo.sif"
            echo "  $(dirname "$REPO_PATH")/empo-gpu.sif"
            exit 1
        fi
    fi
    
    # Verify image exists
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "ERROR: Image not found at $IMAGE_PATH"
        echo "Please build or pull the image first. See CLUSTER.md for instructions."
        exit 1
    fi
fi

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
if [ "$USE_CONTAINER" = "true" ]; then
    echo "  Container runtime: $CONTAINER_CMD"
    echo "  Image: $IMAGE_PATH"
    echo "  Repository: $REPO_PATH"
else
    echo "  Running with native Python (no container)"
fi
echo "=================================================="
echo ""

if [ "$USE_CONTAINER" = "false" ]; then
    # Load modules for native Python (adjust for your HPC system)
    # module load python/3.9
    # module load anaconda3
    
    # Activate conda environment if needed
    # source activate empo
    echo "Note: If using native Python, ensure dependencies are installed."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define Python command based on container usage
if [ "$USE_CONTAINER" = "true" ]; then
    # Run with Singularity/Apptainer
    PYTHON_CMD="$CONTAINER_CMD exec --pwd /workspace -B ${REPO_PATH}:/workspace ${IMAGE_PATH} python /workspace"
    echo "Using containerized Python: $CONTAINER_CMD"
else
    # Run with native Python
    PYTHON_CMD="python"
    echo "Using native Python"
fi

# Run the parameter sweep
echo "Starting parameter sweep..."
$PYTHON_CMD experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py \
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
    $PYTHON_CMD experiments/backward_induction_parameter_sweep/analyze_parameter_sweep.py \
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
