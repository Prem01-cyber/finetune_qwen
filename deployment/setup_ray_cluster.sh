#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-head}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
HEAD_ADDRESS="${HEAD_ADDRESS:-}"
NODE_IP="${NODE_IP:-}"

if [[ -z "${NODE_IP}" ]]; then
  NODE_IP="$(hostname -I | awk '{print $1}')"
fi

echo "Starting Ray node with role=${ROLE} node_ip=${NODE_IP}"

if [[ "${ROLE}" == "head" ]]; then
  ray start \
    --head \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --node-ip-address="${NODE_IP}" \
    --disable-usage-stats
  ray status
  echo "Ray head started at ${NODE_IP}:${RAY_PORT}"
elif [[ "${ROLE}" == "worker" ]]; then
  if [[ -z "${HEAD_ADDRESS}" ]]; then
    echo "HEAD_ADDRESS must be set for worker mode"
    exit 1
  fi
  ray start \
    --address="${HEAD_ADDRESS}:${RAY_PORT}" \
    --node-ip-address="${NODE_IP}" \
    --disable-usage-stats
  ray status
  echo "Ray worker joined ${HEAD_ADDRESS}:${RAY_PORT}"
else
  echo "Unknown role: ${ROLE}. Use 'head' or 'worker'."
  exit 1
fi

