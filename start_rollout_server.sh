#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Start Rollout Server (for worker GPUs)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL_PATH="${MODEL_PATH:-checkpoints/dual_task_v1}"
PORT="${PORT:-5000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-4}"

# Detect GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ Found $GPU_COUNT GPU(s) for rollout generation"
else
    echo "⚠ No GPUs detected, running on CPU"
    GPU_COUNT=0
fi

# Activate environment
source .venv/bin/activate

# Ensure server script exists
if [ ! -f "scripts/rollout_server_multi_gpu.py" ]; then
    echo "→ Creating rollout server script..."
    cat > scripts/rollout_server_multi_gpu.py <<'PYEOF'
"""Multi-GPU rollout generation server."""
import os
import json
import time
import logging
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/dual_task_v1")
NUM_GPUS = torch.cuda.device_count()
MAX_WORKERS = NUM_GPUS if NUM_GPUS > 0 else 1

models = []
tokenizers = []

logger.info(f"Loading model {MODEL_PATH} on {MAX_WORKERS} GPUs...")

def load_model_on_gpu(gpu_id: int):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        mdl = mdl.to(f"cuda:{gpu_id}")
    mdl.eval()
    return tok, mdl

for i in range(MAX_WORKERS):
    gpu_id = i if torch.cuda.is_available() else 0
    tok, mdl = load_model_on_gpu(gpu_id)
    tokenizers.append(tok)
    models.append(mdl)
    logger.info(f"✓ Model loaded on GPU {gpu_id}")

def generate_one(prompt: str, max_new_tokens: int, temperature: float, top_p: float, worker_id: int):
    tok = tokenizers[worker_id]
    mdl = models[worker_id]
    device = next(mdl.parameters()).device
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = mdl.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    prompt_text = tok.decode(inp['input_ids'][0], skip_special_tokens=True)
    return text[len(prompt_text):].strip()

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "num_gpus": NUM_GPUS,
        "workers": MAX_WORKERS,
        "model": MODEL_PATH,
    })

@app.post("/generate_batch")
def generate_batch():
    payload = request.get_json(force=True)
    prompts = payload.get("prompts", [])
    max_new_tokens = int(payload.get("max_new_tokens", 256))
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 0.9))

    if not prompts:
        return jsonify({"error": "No prompts provided"}), 400

    t0 = time.time()
    results = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for i, p in enumerate(prompts):
            worker_id = i % MAX_WORKERS
            f = ex.submit(generate_one, p, max_new_tokens, temperature, top_p, worker_id)
            futs[f] = i
        for f in as_completed(futs):
            idx = futs[f]
            try:
                results[idx] = f.result()
            except Exception as e:
                logger.error(f"Generation failed for prompt {idx}: {e}")
                results[idx] = ""

    elapsed = time.time() - t0
    logger.info(f"Generated {len(prompts)} rollouts in {elapsed:.2f}s ({len(prompts)/elapsed:.1f} rollouts/s)")

    return jsonify({
        "generations": results,
        "elapsed_s": elapsed,
        "count": len(prompts),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
PYEOF
fi

echo ""
echo "=========================================="
echo "Starting Rollout Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Listening: $HOST:$PORT"
echo "GPUs: $GPU_COUNT"
echo "Workers: $WORKERS"
echo "=========================================="
echo ""

export MODEL_PATH="$MODEL_PATH"
exec gunicorn \
    --bind "$HOST:$PORT" \
    --workers "$WORKERS" \
    --timeout 600 \
    --access-logfile logs/distributed/rollout_server_access.log \
    --error-logfile logs/distributed/rollout_server_error.log \
    --log-level info \
    scripts.rollout_server_multi_gpu:app
