#!/bin/bash
# =============================================================================
# Calibrated Agency — Prerequisite / Sanity Check
#
# Verifies that the conda environment, GPU drivers, local model paths,
# dataset access, and vLLM inference all work correctly.
#
# Submit from the project root:
#   sbatch scripts/lrz/00_sanity_check.sh
#
# Output logs: logs/sanity_check_<JOBID>.{out,err}
# =============================================================================

#SBATCH --job-name=ca-sanity-check
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/sanity_check_%j.out
#SBATCH --error=logs/sanity_check_%j.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

echo "============================================================"
echo "Calibrated Agency — Sanity Check"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Project   : $PROJECT_DIR"
echo "Time      : $(date)"
echo "============================================================"

# Activate conda (Miniconda3 installed at ~/miniconda3)
source "$HOME/miniconda3/bin/activate"
conda activate calibrated_agency

echo "Python    : $(python --version)"
echo "Conda env : calibrated_agency"

# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU Info ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Run the prerequisite check
# ---------------------------------------------------------------------------
echo ""
echo "--- Running prerequisite checks ---"
python -m scripts.lrz.check_prerequisites

echo ""
echo "Sanity check completed at $(date)"
