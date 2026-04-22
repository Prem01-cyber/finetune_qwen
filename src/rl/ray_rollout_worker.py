"""
Ray rollout worker for GPU-isolated trajectory generation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List

import ray
import torch

from src.rl.curriculum_manager import TopicState
from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.multi_gpu_rollout_worker import WorkerRuntimeConfig, _load_policy_and_value
from src.rl.rollout_buffer import GAEComputer

logger = logging.getLogger(__name__)


def _trajectory_to_dict(trajectory: Any, gae: GAEComputer) -> Dict[str, Any]:
    advantages, returns = gae.compute_advantages_and_returns(trajectory)

    steps: List[Dict[str, Any]] = []
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    actions: List[int] = []
    old_logprobs: List[float] = []
    old_values: List[float] = []
    rewards: List[float] = []
    dones: List[bool] = []

    for transition in trajectory.transitions:
        state_input_ids = transition.state.input_ids.detach().cpu().tolist()
        state_attention_mask = transition.state.attention_mask.detach().cpu().tolist()
        next_input_ids = transition.next_state.input_ids.detach().cpu().tolist()
        next_attention_mask = transition.next_state.attention_mask.detach().cpu().tolist()

        step = {
            "state": {
                "text": transition.state.text,
                "input_ids": state_input_ids,
                "attention_mask": state_attention_mask,
                "phase": transition.state.phase,
            },
            "action": {
                "token_id": int(transition.action.token_id),
                "log_prob": float(transition.action.log_prob),
                "entropy": float(transition.action.entropy),
            },
            "reward": float(transition.reward),
            "next_state": {
                "text": transition.next_state.text,
                "input_ids": next_input_ids,
                "attention_mask": next_attention_mask,
                "phase": transition.next_state.phase,
            },
            "value": float(transition.value),
            "done": bool(transition.done),
        }
        steps.append(step)

        input_ids.append(state_input_ids)
        attention_mask.append(state_attention_mask)
        actions.append(int(transition.action.token_id))
        old_logprobs.append(float(transition.action.log_prob))
        old_values.append(float(transition.value))
        rewards.append(float(transition.reward))
        dones.append(bool(transition.done))

    return {
        "steps": steps,
        "metadata": dict(trajectory.metadata),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "actions": actions,
        "old_logprobs": old_logprobs,
        "old_values": old_values,
        "rewards": rewards,
        "advantages": [float(x) for x in advantages],
        "returns": [float(x) for x in returns],
        "dones": dones,
    }


@ray.remote(num_gpus=1)
class RolloutWorker:
    """
    Independent rollout worker running on dedicated GPU.
    """

    def __init__(self, config_dict: dict, worker_id: int):
        self.worker_id = int(worker_id)
        self.config_dict = dict(config_dict)
        self.device = torch.device("cuda:0")
        self.gae = GAEComputer(
            gamma=float(self.config_dict.get("gamma", 1.0)),
            gae_lambda=float(self.config_dict.get("gae_lambda", 0.95)),
        )

        runtime_config = WorkerRuntimeConfig(
            base_model=str(self.config_dict["base_model"]),
            reference_questions=list(self.config_dict.get("reference_questions", [])),
            curriculum_checkpoint_dir=str(self.config_dict.get("curriculum_checkpoint_dir", "checkpoints/curriculum/ray_workers")),
            max_question_tokens=int(self.config_dict.get("max_question_tokens", 200)),
            max_solution_tokens=int(self.config_dict.get("max_solution_tokens", 500)),
            temperature=float(self.config_dict.get("temperature", 0.7)),
            top_p=float(self.config_dict.get("top_p", 0.9)),
            consensus_temperature=float(self.config_dict.get("consensus_temperature", 0.7)),
            use_vllm=bool(self.config_dict.get("use_vllm_rollouts", False)),
            vllm_batch_size=int(self.config_dict.get("rollout_batch_size", 16)),
            vllm_tensor_parallel_size=int(self.config_dict.get("vllm_tensor_parallel_size", 1)),
            vllm_gpu_memory_utilization=float(self.config_dict.get("vllm_gpu_memory_utilization", 0.85)),
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        self.policy, self.value, self.tokenizer = _load_policy_and_value(runtime_config.base_model)
        self.policy.eval()
        self.value.eval()

        self.env = CurriculumMathEnvironment(
            policy_model=self.policy,
            value_model=self.value,
            tokenizer=self.tokenizer,
            reference_questions=runtime_config.reference_questions,
            curriculum_checkpoint_dir=runtime_config.curriculum_checkpoint_dir,
            max_question_tokens=runtime_config.max_question_tokens,
            max_solution_tokens=runtime_config.max_solution_tokens,
            temperature=runtime_config.temperature,
            top_p=runtime_config.top_p,
            consensus_temperature=runtime_config.consensus_temperature,
            use_vllm=runtime_config.use_vllm,
            vllm_batch_size=runtime_config.vllm_batch_size,
            vllm_tensor_parallel_size=runtime_config.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=runtime_config.vllm_gpu_memory_utilization,
            base_model_path=runtime_config.base_model,
            parallel_worker_timeout_seconds=float(self.config_dict.get("worker_timeout_seconds", 900.0)),
        )

        logger.info("Ray worker %d initialized on device %s", self.worker_id, self.device)

    def _apply_curriculum_state(self, curriculum_state_dict: Dict[str, Any]) -> None:
        if not curriculum_state_dict:
            return

        manager = self.env.curriculum_manager
        manager.current_iteration = int(curriculum_state_dict.get("iteration", manager.current_iteration))
        manager.current_focus_topics = list(curriculum_state_dict.get("current_focus_topics", manager.current_focus_topics))
        manager.recent_combined_rewards = list(curriculum_state_dict.get("recent_combined_rewards", manager.recent_combined_rewards))

        topics_payload = curriculum_state_dict.get("topics", {})
        if not isinstance(topics_payload, dict):
            return

        for topic, values in topics_payload.items():
            if topic not in manager.topics or not isinstance(values, dict):
                continue
            try:
                manager.topics[topic] = TopicState(**values)
            except Exception:
                continue

    def generate_rollouts(self, num_rollouts: int, curriculum_state_dict: dict) -> List[dict]:
        requested = max(0, int(num_rollouts))
        if requested == 0:
            return []

        self._apply_curriculum_state(dict(curriculum_state_dict or {}))
        trajectories: List[dict] = []

        try:
            with torch.no_grad():
                for _ in range(requested):
                    trajectory = self.env.rollout_trajectory()
                    payload = _trajectory_to_dict(trajectory, self.gae)
                    payload["metadata"]["worker_id"] = self.worker_id
                    trajectories.append(payload)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                logger.error("Ray worker %d OOM during rollout generation: %s", self.worker_id, exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return []
            logger.exception("Ray worker %d runtime failure: %s", self.worker_id, exc)
            return []
        except Exception as exc:
            logger.exception("Ray worker %d unexpected failure: %s", self.worker_id, exc)
            return []
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return trajectories

    def update_policy(self, policy_state_dict: dict):
        if not policy_state_dict:
            return
        self.policy.load_state_dict(policy_state_dict, strict=False)
        self.policy.eval()

    def ping(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "device": str(self.device),
            "cuda_available": bool(torch.cuda.is_available()),
        }
