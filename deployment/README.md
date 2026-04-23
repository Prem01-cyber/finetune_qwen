---
title: Self-Improvement Math Env
emoji: 🧮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: OpenEnv self-play environment for RL on math reasoning.
tags:
  - openenv
  - reinforcement-learning
  - self-play
  - math
  - qwen
---

# Self-Improvement Math Environment

OpenEnv-compliant environment for **Theme #4 — Self-Improvement** of the
Apr 2026 OpenEnv Hackathon. An agent proposes a math problem in response
to a curriculum-selected instruction, submits a solution, and the
environment scores both the question and the solution with:

- **SymPy step verification** — process-level arithmetic check (rubric rule #9).
- **Triple-verifier consensus** — two alternate solutions sampled from a
  reference model; majority vote catches semantic errors the SymPy layer
  cannot (rubric rule #7). Note: the deployed server uses the
  lightweight consensus-based path; training additionally loads
  `Qwen2.5-Math-PRM-7B` in 4-bit for per-step correctness rewards —
  that dependency is intentionally kept out of the deployed image to
  minimise cold-start time and VRAM footprint.
- **Format compliance** — steps and final-answer tagging.
- **Expert-panel modifier** — phased curriculum weighting (pedagogy →
  exploration → expertise) that shifts what "good" means as the agent
  improves.
- **Adaptive curriculum** — per-topic success rate drives topic
  probabilities and target difficulty, implementing Zone-of-Proximal-
  Development selection.

See [`HACKATHON.md`](../HACKATHON.md) in the repo root for the full
mapping to the 22 rubric rules.

## HTTP API

| Method | Path       | Purpose                                                |
|--------|------------|--------------------------------------------------------|
| GET    | `/health`  | Liveness probe (no model load).                        |
| GET    | `/metadata`| Static env metadata (reward components, episode model).|
| POST   | `/reset`   | Start an episode; returns curriculum-selected prompt.  |
| POST   | `/step`    | Submit `{question, solution}`; returns reward breakdown.|
| GET    | `/state`   | Current curriculum snapshot (for dashboards).          |
| POST   | `/close`   | Clean-shutdown hint from the client.                   |

### Minimal client

```python
from src.openenv.client import SelfImprovementMathClient
from src.openenv.models import Action

with SelfImprovementMathClient("http://localhost:7860") as env:
    obs = env.reset()
    print("Challenge:", obs.instruction, "(topic:", obs.topic, ")")

    response = env.step(Action(
        question="Maya buys 3 notebooks at $2.50 each and pays with $10. What is her change?",
        solution="3 × 2.50 = 7.50. 10 - 7.50 = 2.50. Final answer: 2.50",
    ))
    print("Reward:", response.reward)
    print("Breakdown:", response.reward_breakdown)
```

### OpenAPI

The server auto-serves an OpenAPI schema at `/docs` (Swagger UI) and
`/openapi.json`. The schema is the single source of truth - if this
README drifts, trust the schema.

## Running the Space

HF Spaces pulls this folder's `Dockerfile` and sets `PORT=7860`. On
first `/reset` the server lazy-loads the policy checkpoint from
`BASE_MODEL` (default `checkpoints/dual_task_v1`) and the value-head
backbone from `BASE_MODEL_NAME` (default
`Qwen/Qwen2.5-Math-1.5B-Instruct`). This keeps boot fast so Docker's
readiness probe succeeds before CUDA is warm.

Persistent state (curriculum checkpoints, HF model cache) lives under
`/data`, which is the HF Spaces persistent volume.

### Running locally

```bash
docker build -f deployment/Dockerfile -t self-improve-env:dev .
docker run --gpus all --rm -p 8000:7860 \
    -e BASE_MODEL=/opt/ckpt/dual_task_v1 \
    -v "$(pwd)/checkpoints/dual_task_v1:/opt/ckpt/dual_task_v1:ro" \
    self-improve-env:dev
```

Or without Docker:

```bash
pip install -r deployment/requirements.txt
python -m src.openenv.server --host 0.0.0.0 --port 8000
```

## Hardware notes

- **Minimum:** 1× 24 GB GPU (A10G, 3090, 4090). Policy + ValueHead load
  to ~6 GB in bfloat16; the rest is headroom for triple-verifier KV cache.
- **CPU-only:** works for the FastAPI scaffolding, but
  `/step` is impractically slow (minutes per episode). Leave the CPU
  path for smoke tests.
- **Free HF Space tier:** too small for a 1.5B model; use a `t4-small`
  or `a10g-small` tier for live demos.

## Safety / anti-reward-hacking

The environment applies multiple *independent* scoring signals
(rubric rule #8). The reward function cannot be satisfied by format
gaming alone, by matching majority without correctness, or by
arithmetic-only correctness without consensus. In addition:

- Actions are length-bounded by pydantic (`question ≤ 4000`,
  `solution ≤ 8000` chars) to prevent prompt-stuffing.
- `measured_difficulty ∈ [0, 1]` is clamped server-side.
- The consensus majority answer is never revealed to the agent, so the
  policy cannot directly optimize for "be the majority."
- The curriculum refuses to sample already-mastered topics unless
  rate-limited by the ZPD controller.

## License

Apache-2.0. See the repo root `LICENSE` file.
