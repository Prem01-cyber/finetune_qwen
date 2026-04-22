#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up rollout server host..."
"$SCRIPT_DIR/setup_distributed.sh" rollout-server

echo ""
echo "Rollout server host setup complete."
echo "Next: ./start_rollout_server.sh"

