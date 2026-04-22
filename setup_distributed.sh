#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Automated Distributed RL Setup for Any GPU Configuration
###############################################################################

ROLE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Distributed RL Training Setup"
echo "=========================================="

# Detect GPU count
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ Detected $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "⚠ No NVIDIA GPUs detected"
    GPU_COUNT=0
fi

# Create or activate venv
if [ ! -d ".venv" ]; then
    echo "→ Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "→ Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Install distributed packages based on role
if [ "$ROLE" = "worker" ] || [ "$ROLE" = "rollout-server" ]; then
    echo "→ Installing worker dependencies (Flask for rollout server)..."
    pip install flask gunicorn -q
elif [ "$ROLE" = "coordinator" ] || [ "$ROLE" = "trainer" ]; then
    echo "→ Installing coordinator dependencies (requests)..."
    pip install requests -q
else
    echo "→ Installing all distributed dependencies..."
    pip install flask gunicorn requests -q
fi

echo "✓ Dependencies installed"

# Create necessary directories
mkdir -p checkpoints/ppo_training_distributed/curriculum
mkdir -p logs/distributed
mkdir -p data/sft

echo "✓ Directory structure created"

# Role-specific setup
case "$ROLE" in
    worker|rollout-server)
        echo ""
        echo "=========================================="
        echo "ROLLOUT WORKER SETUP COMPLETE"
        echo "=========================================="
        echo ""
        echo "GPU Configuration: $GPU_COUNT GPUs available for rollout generation"
        echo ""
        echo "To start the rollout server:"
        echo "  ./start_rollout_server.sh"
        echo ""
        echo "The server will:"
        echo "  - Load model on all available GPUs"
        echo "  - Listen on port 5000 for rollout requests"
        echo "  - Use parallel generation across GPUs"
        ;;
        
    coordinator|trainer)
        echo ""
        echo "=========================================="
        echo "TRAINING COORDINATOR SETUP COMPLETE"
        echo "=========================================="
        echo ""
        echo "GPU Configuration: $GPU_COUNT GPU(s) available for training"
        echo ""
        echo "To start distributed training:"
        echo "  ./start_distributed_training.sh <ROLLOUT_SERVER_URL>"
        echo ""
        echo "Example:"
        echo "  ./start_distributed_training.sh http://198.53.64.194:5000"
        ;;
        
    *)
        echo ""
        echo "=========================================="
        echo "SETUP COMPLETE"
        echo "=========================================="
        echo ""
        echo "Usage:"
        echo "  On ROLLOUT WORKERS (RTX 5090, etc.):"
        echo "    ./setup_distributed.sh worker"
        echo "    ./start_rollout_server.sh"
        echo ""
        echo "  On TRAINING COORDINATOR (H100, etc.):"
        echo "    ./setup_distributed.sh coordinator"
        echo "    ./start_distributed_training.sh <ROLLOUT_SERVER_URL>"
        echo ""
        echo "  For single-machine training:"
        echo "    ./setup_distributed.sh"
        echo "    python scripts/run_ppo_training_curriculum.py --use-vllm-rollouts"
        ;;
esac

echo ""
echo "✓ Setup complete!"
