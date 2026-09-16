#!/bin/bash
# =============================================================================
# Calibrated Agency — Conda Environment Setup for LRZ AI Systems
#
# Run ONCE on the LRZ login node (NOT inside a SLURM job):
#   bash scripts/lrz/setup_env.sh
#
# What this does:
#   1. Installs Miniconda3 if not already present
#   2. Creates the 'calibrated_agency' conda environment (Python 3.11)
#   3. Installs PyTorch 2.4.0 with CUDA 12.6 support (H100 nodes)
#   4. Installs all remaining project requirements from requirements.txt
#
# After setup, submit jobs with:
#   sbatch scripts/lrz/00_sanity_check.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

ENV_NAME="calibrated_agency"
PYTHON_VERSION="3.11"
MINICONDA_ROOT="$HOME/miniconda3"

echo "============================================================"
echo "Calibrated Agency — LRZ Environment Setup"
echo "Project : $PROJECT_DIR"
echo "Env     : $ENV_NAME (Python $PYTHON_VERSION)"
echo "Time    : $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Install Miniconda3 if not already present
# ---------------------------------------------------------------------------
if [ ! -f "$MINICONDA_ROOT/bin/conda" ]; then
    echo ""
    echo "--- Installing Miniconda3 ---"
    INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINICONDA_ROOT"
    rm -f "$INSTALLER"
    echo "Miniconda3 installed at: $MINICONDA_ROOT"
else
    echo "Miniconda3 already present at: $MINICONDA_ROOT"
fi

# Activate base env (as shown in the LRZ tutorial)
source "$MINICONDA_ROOT/bin/activate"
echo "Conda version : $(conda --version)"

# ---------------------------------------------------------------------------
# 2. Create conda environment
# ---------------------------------------------------------------------------
echo ""
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "--- Conda env '$ENV_NAME' already exists, skipping creation ---"
else
    echo "--- Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION) ---"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
echo "Active env : $ENV_NAME"
echo "Python     : $(python --version)"

# ---------------------------------------------------------------------------
# 3. Install PyTorch 2.4.0 with CUDA 12.6 (for LRZ H100 / A100 nodes)
# ---------------------------------------------------------------------------
echo ""
echo "--- Installing PyTorch 2.4.0 + CUDA 12.6 ---"
pip install torch==2.4.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \
    --quiet

python -c "
import torch
print(f'  PyTorch   : {torch.__version__}')
print(f'  CUDA build: {torch.version.cuda}')
print(f'  Available : {torch.cuda.is_available()}')
"

# ---------------------------------------------------------------------------
# 4. Install all remaining requirements
# ---------------------------------------------------------------------------
echo ""
echo "--- Installing requirements.txt ---"
# requirements.txt deliberately omits torch/torchvision/torchaudio so the
# CUDA 12.6 wheel installed in step 3 is never downgraded to a CPU build.
pip install -r requirements.txt --quiet

# ---------------------------------------------------------------------------
# 5. Verify key imports
# ---------------------------------------------------------------------------
echo ""
echo "--- Verifying imports ---"
python - <<'EOF'
import importlib, sys

checks = [
    ("torch",                 "torch"),
    ("vllm",                  "vllm"),
    ("transformers",          "transformers"),
    ("datasets",              "datasets"),
    ("huggingface_hub",       "huggingface_hub"),
    ("accelerate",            "accelerate"),
    ("sentence_transformers", "sentence_transformers"),
    ("lm_polygraph",          "lm_polygraph"),
    ("sklearn",               "sklearn"),
    ("scipy",                 "scipy"),
    ("mlflow",                "mlflow"),
    ("mcp",                   "mcp"),
    ("rich",                  "rich"),
    ("jsonlines",             "jsonlines"),
    ("dotenv",                "dotenv"),
]

failed = []
for label, module in checks:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "?")
        print(f"  OK   {label:<25} {ver}")
    except ImportError as e:
        print(f"  FAIL {label:<25} {e}")
        failed.append(label)

if failed:
    print(f"\nFailed imports: {failed}")
    sys.exit(1)
else:
    print("\nAll imports OK.")
EOF

# ---------------------------------------------------------------------------
# 6. Copy .env template if not present
# ---------------------------------------------------------------------------
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo ""
    echo "Created .env from template."
    echo "NOTE: No HF_TOKEN needed: models are loaded from local shared storage."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Setup complete at $(date)"
echo ""
echo "Next steps:"
echo "  1. Verify GPU + prerequisites:"
echo "       sbatch scripts/lrz/00_sanity_check.sh"
echo "  2. Run GSM8K baseline (Llama-3.1-8B-Instruct):"
echo "       sbatch scripts/lrz/01_gsm8k_baseline_8b.sh"
echo "  3. Run GSM8K baseline (Llama-3.1-70B-Instruct):"
echo "       sbatch scripts/lrz/02_gsm8k_baseline_70b.sh"
echo ""
echo "Useful SLURM commands:"
echo "  squeue -u \$USER -l          # list your jobs"
echo "  tail -f logs/<job>.out       # stream output"
echo "  scancel <JOBID>              # cancel a job"
echo "  sinfo -p lrz-hgx-h100-94x4  # check H100 node availability"
echo "============================================================"
