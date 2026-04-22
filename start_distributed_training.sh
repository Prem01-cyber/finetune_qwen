#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ROLLOUT_SERVER_URL> [extra args...]"
  echo "Example: $0 http://198.53.64.194:5000 --num-iterations 1000 --rollouts-per-iter 100"
  exit 1
fi

ROLLOUT_SERVER_URL="$1"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run ./setup_distributed.sh coordinator first."
  exit 1
fi

source .venv/bin/activate

python scripts/run_ppo_training_remote_rollouts.py \
  --server-url "$ROLLOUT_SERVER_URL" \
  "$@"

