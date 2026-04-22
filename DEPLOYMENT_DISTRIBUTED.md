# Distributed RL Deployment Guide (vast.ai / runpod)

## 1) Rent Instances

- Coordinator/trainer: 1x H100 (80GB recommended)
- Rollout workers: 8x A10 (24GB each)
- Image: CUDA 12.x + Python 3.10+ + SSH access
- Expose ports on coordinator: `6379` (Ray), `8265` (Ray dashboard), `22` (SSH)

## 2) Install Environment (all nodes)

```bash
git clone <your-repo-url>
cd Finetune_qwen
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "ray[default]==2.30.0" "vllm==0.6.3.post1"
```

## 3) Start Ray Cluster

On coordinator (H100):

```bash
chmod +x deployment/setup_ray_cluster.sh
NODE_IP=<H100_PRIVATE_IP> ./deployment/setup_ray_cluster.sh head
```

On each A10 worker:

```bash
chmod +x deployment/setup_ray_cluster.sh
HEAD_ADDRESS=<H100_PRIVATE_IP> NODE_IP=<WORKER_PRIVATE_IP> ./deployment/setup_ray_cluster.sh worker
```

Verify from coordinator:

```bash
ray status
```

You should see 9 nodes total (1 head + 8 workers).

## 4) Quick 2-worker Sanity Test

On coordinator:

```bash
python scripts/test_distributed_rollouts.py \
  --ray-address auto \
  --num-workers 2 \
  --base-model checkpoints/dual_task_v1 \
  --rollouts 8 \
  --no-gpu-workers
```

This validates worker creation, rollout aggregation, and timing without requiring 8 GPUs.

## 5) Launch Distributed Training

```bash
python scripts/run_ppo_training_distributed.py \
  --base-model checkpoints/dual_task_v1 \
  --output-dir checkpoints/ppo_training_distributed \
  --num-iterations 1000 \
  --rollouts-per-iter 100 \
  --num-workers 8 \
  --ray-address auto \
  --rollout-batch-size 8 \
  --sync-every 1
```

Recommended toggles:

- Disable async buffering for debugging:
  - `--no-async-buffer`
- Disable vLLM workers if debugging fallback path:
  - `--no-vllm-workers`
- Run CPU smoke tests locally:
  - `--no-gpu-workers`

## 6) Monitoring

- Ray dashboard: `http://<H100_PUBLIC_IP>:8265`
- Logs: `tail -f ray/session_latest/logs/*.log`
- Training metrics:
  - `checkpoints/ppo_training_distributed/iteration_*/metrics.json`
- Optional W&B dashboard if enabled.

## 7) Expected Benchmarks

These are practical targets for your setup:

- Single-GPU baseline: ~27 min/iteration
- vLLM + batched rollouts (single node): 5-10 min/iteration
- 8x A10 distributed rollouts + H100 trainer: 2-5 min/iteration
- Async prefetch enabled: training should see near-continuous rollout availability

## 8) Failure Recovery

- Worker failures are retried by coordinator and actors are restarted.
- Resume from latest checkpoint:

```bash
python scripts/run_ppo_training_distributed.py ... --output-dir <existing_dir>
```

- Keep checkpoint cadence aggressive during hackathon:
  - `--save-every 1`

## 9) Common Issues

- **Ray cannot connect**
  - Confirm private IP reachability and open port `6379`.
- **GPU OOM in workers**
  - Reduce `--rollout-batch-size` (e.g. from 8 to 4).
- **vLLM init failure**
  - Ensure CUDA/NVIDIA driver compatibility and `vllm` installation.
- **Slow sync**
  - Use PEFT/LoRA trainable params to reduce synchronization payload.

