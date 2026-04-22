"""
Proximal Policy Optimisation (PPO) Trainer.

Implements the clipped-surrogate objective (Schulman et al., 2017):

    L^PPO(θ) = E_t[ min(r_t(θ)·Â_t,  clip(r_t(θ), 1-ε, 1+ε)·Â_t) ]
               - c1 · L^VF + c2 · H

where:
    r_t(θ)  = π_θ(a_t|s_t) / π_old(a_t|s_t)   probability ratio
    Â_t     = GAE advantage (from RolloutBuffer)
    L^VF    = clipped value-function MSE loss
    H       = mean entropy over the batch
    c1, c2  = vf_coef, ent_coef

Early stopping: if approx KL divergence exceeds target_kl the epoch
is aborted to avoid destructive policy updates.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


class PPOLoss(nn.Module):
    """
    Stateless PPO loss calculator.

    Args:
        clip_range    : ε for policy ratio clipping.
        clip_range_vf : ε for value-function clipping (None = no clipping).
        vf_coef       : Weight c1 for value loss.
        ent_coef      : Weight c2 for entropy bonus.
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        clip_range_vf: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict]:
        """
        Clipped surrogate policy loss.

        Returns:
            loss : scalar tensor
            info : dict with clip_fraction, approx_kl
        """
        ratio = torch.exp(log_probs - old_log_probs)

        # Unclipped and clipped objectives
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        clip_fraction = (
            (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
        )
        approx_kl = (
            ((ratio - 1.0) - torch.log(ratio)).mean().item()
        )

        return policy_loss, {"clip_fraction": clip_fraction, "approx_kl": approx_kl}

    def compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped value-function MSE loss."""
        vf_loss_unclipped = (values - returns) ** 2
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_range_vf, self.clip_range_vf
        )
        vf_loss_clipped = (values_clipped - returns) ** 2
        return 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()


class PPOTrainer:
    """
    Orchestrates PPO updates for policy and value networks.

    Args:
        policy_model  : Language model π_θ (AutoModelForCausalLM).
        value_model   : Critic V_φ (ValueHead).
        tokenizer     : Tokenizer (for saving).
        learning_rate : Adam learning rate.
        ppo_epochs    : Number of gradient epochs over each rollout buffer.
        batch_size    : Mini-batch size (in transitions).
        clip_range    : PPO ε.
        clip_range_vf : Value-function clip ε.
        vf_coef       : Value loss coefficient c1.
        ent_coef      : Entropy bonus coefficient c2.
        max_grad_norm : Gradient clipping norm.
        target_kl     : Early-stopping KL threshold.
    """

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
    ) -> None:
        self.policy = policy_model
        self.value = value_model
        self.tokenizer = tokenizer

        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.loss_fn = PPOLoss(
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
        )

        # Single optimiser covers both policy LoRA params and value head MLP
        trainable_params = list(
            filter(lambda p: p.requires_grad, policy_model.parameters())
        ) + list(value_model.value_head.parameters())

        self.optimiser = AdamW(trainable_params, lr=learning_rate)

        self.device = next(policy_model.parameters()).device

    def _policy_logits_at_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Logits for the next-token distribution at the last non-pad position.

        Returns:
            logits_last: [batch, vocab_size] float32
        """
        out = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = out.logits.float()
        last_idx = attention_mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        b = torch.arange(logits.size(0), device=logits.device)
        return logits[b, last_idx]

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run *ppo_epochs* of gradient updates over the rollout buffer.

        Returns:
            Dict with policy_loss, value_loss, entropy, approx_kl, clip_fraction.
        """
        self.policy.train()
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

        # Every mini-batch runs a full fwd+bwd on the 1.5B policy *and* the
        # value head (~150 ms on an A100), so a typical update does a few
        # hundred steps.  A single bar across ppo_epochs * batches_per_epoch
        # is the right granularity — per-epoch bars thrash the terminal and
        # per-iteration log lines hide the shape of the early-KL exit.
        batches_per_epoch = max(1, math.ceil(len(rollout_buffer) / self.batch_size))
        total_steps = self.ppo_epochs * batches_per_epoch
        pbar = tqdm(
            total=total_steps,
            desc="PPO update",
            unit="step",
            dynamic_ncols=True,
            leave=False,
        )

        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            for batch in rollout_buffer.get_batches(
                batch_size=self.batch_size, shuffle=True
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                action_token_ids = batch["action_token_ids"].to(self.device)

                old_log_probs = batch["old_log_probs"].to(self.device).detach()
                old_values = batch["old_values"].to(self.device).detach()
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # --- policy: re-forward π_θ(a|s) at rollout state ---
                logits_last = self._policy_logits_at_state(input_ids, attention_mask)
                log_probs = F.log_softmax(logits_last, dim=-1)
                new_log_probs = log_probs[
                    torch.arange(logits_last.size(0), device=logits_last.device),
                    action_token_ids,
                ]
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

                policy_loss, pg_info = self.loss_fn.compute_policy_loss(
                    new_log_probs, old_log_probs, advantages
                )

                if pg_info["approx_kl"] > 1.5 * self.target_kl:
                    logger.info(
                        f"Early stopping at epoch {epoch}: "
                        f"approx_kl={pg_info['approx_kl']:.4f}"
                    )
                    early_stopped = True
                    break

                # --- value: re-forward V_φ(s) ---
                new_values = self.value(input_ids, attention_mask).float().squeeze(-1)
                value_loss = self.loss_fn.compute_value_loss(
                    new_values, old_values, returns
                )

                mean_entropy = entropy.mean()

                total_loss = (
                    policy_loss
                    + self.loss_fn.vf_coef * value_loss
                    - self.loss_fn.ent_coef * mean_entropy
                )

                self.optimiser.zero_grad()
                total_loss.backward()
                _params: list[torch.nn.Parameter] = []
                for g in self.optimiser.param_groups:
                    _params.extend(g["params"])
                nn.utils.clip_grad_norm_(_params, self.max_grad_norm)
                self.optimiser.step()
                update_steps += 1

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(mean_entropy.item())
                stats["approx_kl"].append(pg_info["approx_kl"])
                stats["clip_fraction"].append(pg_info["clip_fraction"])

                pbar.set_postfix(
                    ep=f"{epoch + 1}/{self.ppo_epochs}",
                    pl=f"{policy_loss.item():+.4f}",
                    vl=f"{value_loss.item():.4f}",
                    kl=f"{pg_info['approx_kl']:.4f}",
                    clip=f"{pg_info['clip_fraction']:.2f}",
                    refresh=False,
                )
                pbar.update(1)

        pbar.close()

        if update_steps == 0:
            logger.warning(
                "PPO train_step performed 0 optimizer updates. "
                "Likely immediate KL early-stop (target_kl=%.4f).",
                self.target_kl,
            )

        metrics = {k: float(sum(v) / max(len(v), 1)) for k, v in stats.items()}
        metrics["update_steps"] = float(update_steps)
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save policy, value head, and optimiser state.

        Args:
            path : File path ending in .pt
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "value_head_state_dict": self.value.value_head.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

        # Save policy separately (HuggingFace format)
        policy_dir = os.path.join(os.path.dirname(path), "policy")
        self.policy.save_pretrained(policy_dir)
        self.tokenizer.save_pretrained(policy_dir)
        logger.info(f"Policy saved to {policy_dir}")

    def load_checkpoint(self, path: str) -> None:
        """Load value head and optimiser state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.value.value_head.load_state_dict(
            checkpoint["value_head_state_dict"]
        )
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")
