#!/usr/bin/env bash
# =============================================================================
# AgentBench Setup Script (LRZ-compatible with Enroot)
# Run from the project root: bash scripts/setup_agentbench.sh
#
# Only the DB environment (AgentBench-DB, SQL interaction) is used in the final
# thesis. The OS environment (bash/Docker interaction) was dropped from the final
# scope; the Docker/Enroot image pulls below are only needed if you opt into
# --use-docker mode (default is --use-python-api, which needs neither); see
# legacy/README.md for the archived AgentBench-OS-only tooling.
# =============================================================================
set -euo pipefail

echo "=== AgentBench Setup ==="

# Clone AgentBench
if [ ! -d "AgentBench" ]; then
    echo "Cloning AgentBench..."
    git clone https://github.com/THUDM/AgentBench.git
else
    echo "AgentBench directory already exists, skipping clone."
fi

# Install AgentBench dependencies
echo "Installing AgentBench dependencies..."
pip install -r AgentBench/requirements.txt --quiet

# Optional: Set up container images for containerized benchmarks
# Choose one method based on your environment:

if command -v docker &> /dev/null; then
    # Use Docker if available
    echo "Docker detected. Pulling Docker images (this may take a while)..."
    docker pull thudm/agentbench:os
    docker pull thudm/agentbench:db
    echo "Verifying OS environment..."
    docker run --rm thudm/agentbench:os echo "OS environment OK"
    echo "Verifying DB environment..."
    docker run --rm thudm/agentbench:db echo "DB environment OK"
elif command -v enroot &> /dev/null; then
    # Use Enroot (for LRZ and similar HPC systems)
    echo "Docker not found, using Enroot instead..."
    echo "Importing AgentBench images via Enroot..."
    enroot import docker://thudm/agentbench:os
    enroot import docker://thudm/agentbench:db
    echo "Enroot images imported. Run on compute nodes with:"
    echo "  enroot create thudm/agentbench:os"
    echo "  enroot start thudm/agentbench:os echo 'OS environment OK'"
else
    echo "Warning: Docker and Enroot not found."
    echo "Containerized benchmarks (OS/DB tasks) will not be available."
    echo "You can still run LLM-as-agent benchmarks without containers."
fi

echo ""
echo "=== AgentBench setup complete ==="
echo "Python dependencies installed. Ready to run benchmarks."
echo ""
echo "Task data expected at:"
echo "  AgentBench/data/os_interaction/data/ (OS tasks)"
echo "  AgentBench/data/dbbench/ (DB tasks - dev.jsonl or standard.jsonl)"
echo ""
echo "To get started with CalibratedAgency experiments:"
echo "  cd .."
echo "  python -m experiments.agentbench_agent --env-type db --n-samples 10   # final thesis benchmark"
echo "  python -m experiments.agentbench_agent --env-type os --n-samples 10   # excluded from final thesis"
