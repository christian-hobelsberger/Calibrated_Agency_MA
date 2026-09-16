#!/usr/bin/env bash
# =============================================================================
# ToolBench Setup Script
# Run from the project root: bash scripts/setup_toolbench.sh
#
# For offline experiments, the mock server replays cached API responses.
# =============================================================================
set -euo pipefail

echo "=== ToolBench Setup ==="

# Clone ToolBench
if [ ! -d "ToolBench" ]; then
    echo "Cloning ToolBench..."
    git clone https://github.com/OpenBMB/ToolBench.git
else
    echo "ToolBench directory already exists, skipping clone."
fi

# Install ToolBench dependencies
echo "Installing ToolBench dependencies..."
pip install -r ToolBench/requirements.txt --quiet

echo ""
echo "=== Starting mock tool server (background) ==="
# The mock server replays cached API responses, no real API key required.
if [ -f "ToolBench/toolbench/tooleval/tool_server.py" ]; then
    python ToolBench/toolbench/tooleval/tool_server.py \
        --tool_root_dir ToolBench/data/toolenv/tools &
    TOOL_SERVER_PID=$!
    echo "Mock server started (PID $TOOL_SERVER_PID) on http://localhost:5000"
    echo "To stop: kill $TOOL_SERVER_PID"
else
    echo "WARNING: tool_server.py not found. Check ToolBench repo structure."
    echo "Expected: ToolBench/toolbench/tooleval/tool_server.py"
fi

echo ""
echo "=== ToolBench setup complete ==="
echo "Query data expected at:"
echo "  ToolBench/data/test_instructions/G1_instruction.json"
echo "  ToolBench/data/test_instructions/G2_instruction.json"
echo ""
echo "If using real APIs, export RAPIDAPI_KEY=<your_key> before running."
