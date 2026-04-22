"""
Flask-based rollout generation server for multi-GPU remote workers.
"""

import os
import json
import time
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/dual_task_v1")
NUM_GPUS = torch.cuda.device_count()
MAX_WORKERS = NUM_GPUS if NUM_GPUS > 0 else 1

models = []
tokenizers = []

print(f"Loading models on {NUM_GPUS} GPUs...")

def load_model_on_gpu(gpu_id: int):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
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
    print(f"Loading model on GPU {gpu_id}...")
    tok, mdl = load_model_on_gpu(gpu_id)
    tokenizers.append(tok)
    models.append(mdl)

print(f"All {MAX_WORKERS} workers ready")

def generate_one(prompt: str, max_new_tokens: int, temperature: float, top_p: float, worker_id: int):
    tok = tokenizers[worker_id]
    mdl = models[worker_id]
    device = next(mdl.parameters()).device
    
    inp = tok(prompt, return_tensors="pt").to(device)
    
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
    return text

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
            results[idx] = f.result()
    
    return jsonify({
        "generations": results,
        "elapsed_s": time.time() - t0,
        "count": len(prompts),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
