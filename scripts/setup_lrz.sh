#!/usr/bin/env bash
# =============================================================================
# LRZ AI Systems Setup Script
# Run on the LRZ cluster login node:
#   bash scripts/setup_lrz.sh
# =============================================================================
set -euo pipefail

echo "=== Calibrated Agency — LRZ Setup ==="

# --- Load modules ---
module load python/3.11
module load cuda/12.4
module load gcc/13.2

echo "Python: $(python --version)"
echo "CUDA: $(nvcc --version | head -1)"

# --- Create virtual environment ---
ENV_DIR="$HOME/envs/calibrated_agency"
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment at $ENV_DIR..."
    python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# --- Install PyTorch for CUDA 12.4 ---
echo "Installing PyTorch..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 --quiet

# --- Install all other requirements ---
echo "Installing project requirements..."
pip install -r requirements.txt --quiet

# --- Verify GPU ---
echo ""
echo "=== GPU Verification ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# --- Verify vLLM ---
echo ""
echo "=== vLLM Verification ==="
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# --- Verify lm-polygraph ---
python -c "import lm_polygraph; print('lm-polygraph OK')"

# --- Copy environment template ---
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from template. Edit it to add your HF_TOKEN."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate environment with: source $ENV_DIR/bin/activate"
echo "Then run e.g.: python -m experiments.gsm8k_agent --n-samples 10 --split-phase full"
