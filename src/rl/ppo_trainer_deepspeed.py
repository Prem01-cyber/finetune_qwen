"""
DeepSpeed-backed PPO trainer for memory-efficient multi-GPU updates.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


class PPOTrainerDeepSpeed:
    """PPO trainer that shards policy/value training with DeepSpeed ZeRO-3."""

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,
        value_model: ValueHead,
        tokenizer: AutoTokenizer,
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        clip_range: float = 0.2,
        clip_range_vf: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        target_kl: float = 0.01,
        ds_config: Optional[str] = None,
    ) -> None:

        self.tokenizer = tokenizer
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm

        config_path = ds_config or "configs/deepspeed_zero3_rl.json"
        with open(config_path, "r", encoding="utf-8") as handle:
            ds_cfg = json.load(handle)

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        micro_batch = max(1, batch_size // max(world_size, 1))
        ds_cfg["train_batch_size"] = int(batch_size)
        ds_cfg["train_micro_batch_size_per_gpu"] = int(micro_batch)
        ds_cfg["gradient_accumulation_steps"] = 1

        # Only wrap policy in DeepSpeed (it's 1.5B params)
        # Keep value network in standard PyTorch (it's tiny ~few MB)
        policy_params = [p for p in policy_model.parameters() if p.requires_grad]
        
        logger.info("Initializing policy DeepSpeed engine (ZeRO-3)")
        self.policy_engine, self.policy_optimizer, _, _ = deepspeed.initialize(
            model=policy_model,
            model_parameters=policy_params,
            config=ds_cfg,
        )
        
        # Move value network to same device as policy
        self.value = value_model.to(self.policy_engine.device)
        self.value_optimizer = torch.optim.AdamW(
            [p for p in value_model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        self.policy = self.policy_engine.module
        self.device = self.policy_engine.device
        self._last_checkpoint_meta: Dict[str, str] = {}

        logger.info(
            "DeepSpeed ready: world_size=%d, micro_batch_per_gpu=%d",
            world_size,
            micro_batch,
        )

    def _policy_logits_at_state(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Must use the engine (not .module) so ZeRO-3 gathers sharded params
        # and builds a proper computation graph for backward
        outputs = self.policy_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits.float()
        last_idx = attention_mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        row_idx = torch.arange(logits.size(0), device=logits.device)
        return logits[row_idx, last_idx]

    def _value_estimates(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.value(input_ids=input_ids, attention_mask=attention_mask).float()

    def train_step(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        self.policy_engine.train()
        self.value.train()

        stats: Dict[str, list] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        early_stopped = False
        update_steps = 0

        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            for batch in rollout_buffer.get_batches(batch_size=self.batch_size, shuffle=True):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                action_token_ids = batch["action_token_ids"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device).detach()
                old_values = batch["old_values"].to(self.device).detach()
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                logits_last = self._policy_logits_at_state(input_ids, attention_mask)
                log_probs = F.log_softmax(logits_last, dim=-1)
                new_log_probs = log_probs[
                    torch.arange(logits_last.size(0), device=logits_last.device),
                    action_token_ids,
                ]
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()

                if approx_kl > 1.5 * self.target_kl:
                    logger.info(
                        "Early stopping at epoch %d: approx_kl=%.4f",
                        epoch,
                        approx_kl,
                    )
                    early_stopped = True
                    break

                new_values = self._value_estimates(input_ids, attention_mask).squeeze(-1)
                vf_loss_unclipped = (new_values - returns) ** 2
                values_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.clip_range_vf, self.clip_range_vf
                )
                vf_loss_clipped = (values_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

                # Policy update (DeepSpeed ZeRO-3 with offload)
                policy_objective = policy_loss - self.ent_coef * entropy
                
                # DeepSpeed engine handles gradient zeroing automatically
                # Just do backward on the loss, then step the engine
                policy_objective.backward()
                self.policy_engine.step()
                
                # Value update (standard PyTorch)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                update_steps += 1

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_fraction"].append(clip_fraction)

        if update_steps == 0:
            logger.warning(
                "PPO train_step performed 0 optimizer updates. target_kl=%.4f",
                self.target_kl,
            )

        metrics = {key: float(sum(values) / max(len(values), 1)) for key, values in stats.items()}
        metrics["update_steps"] = float(update_steps)
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """
        Save sharded DeepSpeed checkpoints while preserving existing manager API.
        """
        iteration_dir = os.path.dirname(path)
        os.makedirs(iteration_dir, exist_ok=True)

        policy_dir = os.path.join(iteration_dir, "policy_ds")
        
        self.policy_engine.save_checkpoint(policy_dir)
        self.tokenizer.save_pretrained(os.path.join(iteration_dir, "policy_tokenizer"))
        
        # Save value network (standard PyTorch)
        value_checkpoint = {
            "value_head_state_dict": self.value.value_head.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
        }
        torch.save(value_checkpoint, os.path.join(iteration_dir, "value.pt"))

        meta = {
            "format": "deepspeed_zero3",
            "policy_dir": policy_dir,
        }
        torch.save(meta, path)
        self._last_checkpoint_meta = meta
        logger.info("DeepSpeed checkpoint saved under %s", iteration_dir)

    def load_checkpoint(self, path: str) -> None:
        """Restore sharded DeepSpeed checkpoint."""
        if os.path.exists(path):
            meta = torch.load(path, map_location="cpu")
            policy_dir = meta.get("policy_dir")
        else:
            iteration_dir = os.path.dirname(path)
            policy_dir = os.path.join(iteration_dir, "policy_ds")

        self.policy_engine.load_checkpoint(policy_dir)
        
        # Load value network (standard PyTorch)
        value_path = os.path.join(os.path.dirname(path), "value.pt")
        if os.path.exists(value_path):
            checkpoint = torch.load(value_path, map_location=self.device)
            self.value.value_head.load_state_dict(checkpoint["value_head_state_dict"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        
        logger.info("DeepSpeed checkpoint loaded from %s", os.path.dirname(path))
