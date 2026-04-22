"""FastAPI server exposing ``SelfImprovementMathEnv`` over HTTP.

Rubric rule #13 ("Deploy your environment early") and rule #5 ("FastAPI
wrapper / client-server interface") both point at this file.  It does
one thing: lift the in-process env into a process-isolated HTTP
service so

* judges can hit a live URL,
* teammates can train against the same server,
* and the same image runs locally (``uvicorn``) or in a Hugging Face
  Space (Docker).

The model is loaded lazily on the first *real* request so that
``uvicorn`` boot is fast and ``/health`` can pass Docker's readiness
probe before CUDA is warm.

Run locally:

    python -m src.openenv.server \\
        --base-model checkpoints/dual_task_v1 \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.openenv.environment import SelfImprovementMathEnv, _NoPendingEpisodeError
from src.openenv.models import (
    Observation,
    ResetRequest,
    StateResponse,
    StepRequest,
    StepResponse,
)
from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime config (env vars or CLI)
# ---------------------------------------------------------------------------
class ServerConfig:
    """Shared config filled by ``main()`` or the environment."""

    base_model: str = os.environ.get("BASE_MODEL", "checkpoints/dual_task_v1")
    base_model_name: str = os.environ.get(
        "BASE_MODEL_NAME", "Qwen/Qwen2.5-Math-1.5B-Instruct"
    )
    device: str = os.environ.get(
        "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    curriculum_dir: str = os.environ.get(
        "CURRICULUM_CHECKPOINT_DIR", "checkpoints/curriculum_openenv"
    )
    max_question_tokens: int = int(os.environ.get("MAX_QUESTION_TOKENS", "200"))
    max_solution_tokens: int = int(os.environ.get("MAX_SOLUTION_TOKENS", "500"))


_ENV: Optional[SelfImprovementMathEnv] = None


def _build_env(cfg: ServerConfig) -> SelfImprovementMathEnv:
    """Construct the policy + value-head and hand them to the curriculum env.

    ``CurriculumMathEnvironment`` wires up the triple-verifier,
    consensus reward calculator, question-quality evaluator, curriculum
    manager, replay buffer and expert panel on its own - we just
    provide the model and tokenizer.
    """

    logger.info("Loading tokenizer from %s", cfg.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_dir = Path(cfg.base_model)
    logger.info("Loading policy (bfloat16) from %s", cfg.base_model)
    if (adapter_dir / "adapter_config.json").is_file():
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        policy = PeftModel.from_pretrained(base, adapter_dir)
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    policy = policy.to(cfg.device).eval()

    logger.info("Loading value head backbone from %s", cfg.base_model_name)
    value = ValueHead(
        base_model_path=cfg.base_model_name,
        freeze_backbone=True,
        model_device_map=None,
    ).to(cfg.device).eval()

    curriculum_env = CurriculumMathEnvironment(
        policy_model=policy,
        value_model=value,
        tokenizer=tokenizer,
        curriculum_checkpoint_dir=cfg.curriculum_dir,
        max_question_tokens=cfg.max_question_tokens,
        max_solution_tokens=cfg.max_solution_tokens,
        device=torch.device(cfg.device),
    )
    return SelfImprovementMathEnv(curriculum_env)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lazy model load on first /reset; cheap boot for healthchecks."""
    global _ENV
    _ENV = None
    yield
    if _ENV is not None:
        _ENV.close()


app = FastAPI(
    title="Self-Improvement Math Environment",
    description=(
        "OpenEnv-compliant environment for Theme #4 (Self-Improvement).  "
        "Agents propose a math problem from a curriculum-selected "
        "instruction, submit a solution, and the env scores it via "
        "triple-verifier consensus + SymPy step-level verification + "
        "expert-panel curriculum modifier."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _env() -> SelfImprovementMathEnv:
    global _ENV
    if _ENV is None:
        _ENV = _build_env(ServerConfig)
    return _ENV


@app.get("/health")
def health() -> dict:
    """Cheap readiness probe - does not trigger model load."""
    return {"status": "ok", "model_loaded": _ENV is not None}


@app.get("/metadata")
def metadata() -> dict:
    return SelfImprovementMathEnv.metadata


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest | None = None) -> Observation:
    req = req or ResetRequest()
    return _env().reset(seed=req.seed, requested_topic=req.requested_topic)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    try:
        result = _env().step(req.action)
    except _NoPendingEpisodeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StepResponse(
        observation=result["observation"],
        reward=result["reward"],
        done=result["done"],
        reward_breakdown=result["reward_breakdown"],
        info=result["info"],
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return _env().state()


@app.post("/close")
def close() -> dict:
    _env().close()
    return {"closed": True}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--base-model", default=ServerConfig.base_model)
    parser.add_argument("--base-model-name", default=ServerConfig.base_model_name)
    parser.add_argument("--device", default=ServerConfig.device)
    parser.add_argument("--curriculum-dir", default=ServerConfig.curriculum_dir)
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    ServerConfig.base_model = args.base_model
    ServerConfig.base_model_name = args.base_model_name
    ServerConfig.device = args.device
    ServerConfig.curriculum_dir = args.curriculum_dir

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
