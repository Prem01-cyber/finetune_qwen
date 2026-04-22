"""
Ray-backed PPO trainer for distributed rollout collection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ray
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.rl.mdp_components import Action, State, Trajectory, Transition
from src.rl.ppo_trainer import PPOTrainer
from src.rl.ray_rollout_worker import RolloutWorker
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


def _dict_to_trajectory(payload: Dict[str, Any]) -> Trajectory:
    trajectory = Trajectory()
    trajectory.metadata = dict(payload.get("metadata", {}))

    steps = payload.get("steps", [])
    for step in steps:
        state_dict = step["state"]
        next_state_dict = step["next_state"]
        action_dict = step["action"]

        state = State(
            text=str(state_dict.get("text", "")),
            input_ids=torch.tensor(state_dict["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(state_dict["attention_mask"], dtype=torch.long),
            phase=str(state_dict.get("phase", "unknown")),
        )
        next_state = State(
            text=str(next_state_dict.get("text", "")),
            input_ids=torch.tensor(next_state_dict["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(next_state_dict["attention_mask"], dtype=torch.long),
            phase=str(next_state_dict.get("phase", "unknown")),
        )
        action = Action(
            token_id=int(action_dict["token_id"]),
            log_prob=float(action_dict["log_prob"]),
            entropy=float(action_dict.get("entropy", 0.0)),
        )

        trajectory.add(
            Transition(
                state=state,
                action=action,
                reward=float(step["reward"]),
                next_state=next_state,
                value=float(step["value"]),
                done=bool(step["done"]),
            )
        )
    return trajectory


class PPOTrainerRay:
    """
    PPO trainer that collects rollouts in parallel Ray workers.
    """

    def __init__(
        self,
        config: Any,
        policy_model: PreTrainedModel,
        value_model: ValueHead,
        tokenizer: PreTrainedTokenizer,
        reference_questions: Optional[List[str]] = None,
        num_workers: int = 2,
    ):
        self.config = config
        self.policy = policy_model
        self.value = value_model
        self.tokenizer = tokenizer
        self.num_workers = max(1, int(num_workers))
        self.latest_trajectories: List[Trajectory] = []

        if not ray.is_initialized():
            ray.init(
                num_gpus=self.num_workers,
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
            )

        worker_cfg = dict(config.__dict__)
        worker_cfg["reference_questions"] = list(reference_questions or [])
        worker_cfg["use_multi_gpu_rollouts"] = False
        worker_cfg["use_vllm_rollouts"] = bool(getattr(config, "use_vllm_rollouts", False))

        self.workers = [
            RolloutWorker.remote(config_dict=worker_cfg, worker_id=i)
            for i in range(self.num_workers)
        ]

        self.central_trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            tokenizer=tokenizer,
            learning_rate=config.learning_rate,
            ppo_epochs=config.ppo_epochs,
            batch_size=config.batch_size,
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range_vf,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
        )

        self._sync_workers()

    def collect_rollouts(
        self,
        num_rollouts: int,
        curriculum_state_dict: Dict[str, Any],
    ) -> RolloutBuffer:
        total = max(0, int(num_rollouts))
        if total == 0:
            self.latest_trajectories = []
            return RolloutBuffer(
                gamma=float(self.config.gamma),
                gae_lambda=float(self.config.gae_lambda),
                pad_token_id=int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0),
            )

        base = total // len(self.workers)
        remainder = total % len(self.workers)
        work_split = [base + (1 if i < remainder else 0) for i in range(len(self.workers))]

        futures = []
        for worker, count in zip(self.workers, work_split):
            if count <= 0:
                continue
            futures.append(
                worker.generate_rollouts.remote(
                    num_rollouts=count,
                    curriculum_state_dict=dict(curriculum_state_dict),
                )
            )

        worker_results = ray.get(futures) if futures else []
        serialized_trajectories: List[Dict[str, Any]] = []
        for partial in worker_results:
            if not partial:
                continue
            serialized_trajectories.extend(partial)

        trajectories = [_dict_to_trajectory(item) for item in serialized_trajectories]
        self.latest_trajectories = trajectories

        rollout_buffer = RolloutBuffer(
            gamma=float(self.config.gamma),
            gae_lambda=float(self.config.gae_lambda),
            pad_token_id=int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0),
        )
        for trajectory in trajectories:
            rollout_buffer.add_trajectory(trajectory)

        return rollout_buffer

    def train_step(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        metrics = self.central_trainer.train_step(rollout_buffer)
        self._sync_workers()
        return metrics

    def _sync_workers(self) -> None:
        policy_state_dict = {
            key: value.detach().cpu()
            for key, value in self.central_trainer.policy.state_dict().items()
        }
        state_ref = ray.put(policy_state_dict)
        ray.get([worker.update_policy.remote(state_ref) for worker in self.workers])

    def save_checkpoint(self, path: str) -> None:
        self.central_trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.central_trainer.load_checkpoint(path)
        self._sync_workers()

    def shutdown(self) -> None:
        try:
            ray.get([worker.ping.remote() for worker in self.workers], timeout=5.0)
        except Exception:
            pass
