#!/bin/bash
# =============================================================================
# Calibrated Agency — GSM8K Baseline: Llama-3.1-70B-Instruct
#
# Runs the full GSM8K experiment (500 problems) with the 70B model.
# Uses step-wise CoCoA (M=5) and branching consistency (M=5) UQ scoring.
#
# Resources: 4 × H100 94 GB = 376 GB total VRAM
#   70B bfloat16 weights ≈ 140 GB; tensor_parallel_size=4 splits across all 4.
#   Remaining VRAM per GPU is used by vLLM for the KV-cache.
#   NOTE: tensor_parallel_size must always equal --gres=gpu:N.
#
# Estimated wall time: ~18–30 hours for 500 problems.
#
# Submit from the project root:
#   sbatch scripts/lrz/02_gsm8k_baseline_70b.sh
#
# Quick pilot (25 problems):
#   sbatch --export=ALL,N_SAMPLES=25 scripts/lrz/02_gsm8k_baseline_70b.sh
#
# Output logs: logs/gsm8k_70b_<JOBID>.{out,err}
# Results   : results/gsm8k/trajectories.jsonl  (override with OUTPUT_DIR=...)
# =============================================================================

#SBATCH --job-name=ca-gsm8k-70b
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/gsm8k_70b_%j.out
#SBATCH --error=logs/gsm8k_70b_%j.err

# ---------------------------------------------------------------------------
# Configuration — override via: sbatch --export=ALL,VAR=value ...
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-${LRZ_MODEL_DIR:-/path/to/models}/Llama-3.1-70B-Instruct}"
N_SAMPLES="${N_SAMPLES:-500}"
# Default to a 70B-tagged subdir so 8B and 70B runs don't overwrite each other.
OUTPUT_DIR="${OUTPUT_DIR:-results/gsm8k_70b}"
COCOA_M="${COCOA_M:-5}"
BRANCH_M="${BRANCH_M:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"    # must match --gres=gpu:3 above

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs "$OUTPUT_DIR"

echo "============================================================"
echo "Calibrated Agency — GSM8K Baseline (Llama-3.1-70B-Instruct)"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Project   : $PROJECT_DIR"
echo "Model     : $MODEL_PATH"
echo "N samples : $N_SAMPLES"
echo "Output    : $OUTPUT_DIR"
echo "CoCoA M   : $COCOA_M   Branch M : $BRANCH_M"
echo "Temp      : $TEMPERATURE   TP size : $TENSOR_PARALLEL"
echo "Time      : $(date)"
echo "============================================================"

# Activate conda
source "$HOME/miniconda3/bin/activate"
conda activate calibrated_agency
echo "Python    : $(python --version)"

# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU Info ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# ---------------------------------------------------------------------------
# Verify model path and GPU count
# ---------------------------------------------------------------------------
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi
echo ""
echo "Model path verified: $MODEL_PATH"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$N_GPUS" -lt "$TENSOR_PARALLEL" ]; then
    echo "ERROR: tensor_parallel_size=$TENSOR_PARALLEL but only $N_GPUS GPU(s) allocated."
    echo "       Ensure --gres=gpu:$TENSOR_PARALLEL matches TENSOR_PARALLEL."
    exit 1
fi
echo "GPUs allocated: $N_GPUS  (tensor_parallel_size=$TENSOR_PARALLEL)"

# ---------------------------------------------------------------------------
# Run experiment (resumes automatically if interrupted)
# ---------------------------------------------------------------------------
echo ""
echo "--- Starting GSM8K experiment ---"
echo "Start: $(date)"

# Capture exit code under `set -e` by short-circuiting with `||`.
EXIT_CODE=0
python -m experiments.gsm8k_agent \
    --model                "$MODEL_PATH" \
    --n-samples            "$N_SAMPLES" \
    --output-dir           "$OUTPUT_DIR" \
    --cocoa-m              "$COCOA_M" \
    --branch-m             "$BRANCH_M" \
    --temperature          "$TEMPERATURE" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    || EXIT_CODE=$?

echo ""
echo "End: $(date)   Exit code: $EXIT_CODE"

# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------
TRAJ="$OUTPUT_DIR/trajectories.jsonl"
if [ -f "$TRAJ" ]; then
    N_DONE=$(wc -l < "$TRAJ")
    echo ""
    echo "=== Results ==="
    echo "Completed : $N_DONE / $N_SAMPLES"
    echo "File      : $TRAJ"
    python - "$OUTPUT_DIR" <<'PYEOF'
import json, pathlib, statistics, sys

traj = pathlib.Path(sys.argv[1]) / "trajectories.jsonl"
records = [json.loads(l) for l in traj.open() if l.strip()]
if records:
    acc = sum(r["task_success"] for r in records) / len(records)
    steps = [r.get("n_steps", 0) for r in records]
    print(f"Accuracy          : {acc:.3f}  ({sum(r['task_success'] for r in records)}/{len(records)})")
    print(f"Avg steps/problem : {statistics.mean(steps):.1f}")
PYEOF
fi

# Propagate the experiment's exit code so SLURM marks the job correctly.
exit $EXIT_CODE
