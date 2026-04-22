"""
DeepSpeed-backed PPO trainer for memory-efficient multi-GPU updates.

Design
------
This trainer keeps the clipped-surrogate PPO objective of ``ppo_trainer.py``
but wraps **both** the policy and the critic in independent DeepSpeed ZeRO-3
engines.  Why this layout:

* ZeRO-3 shards model params, gradients *and* optimizer state across every
  data-parallel rank.  On our 1.5B policy this turns a ~24GB training memory
  footprint into a few GB per GPU, which is the whole reason we bother with
  DeepSpeed here.

* The value network is a frozen 1.5B backbone + a tiny MLP head.  We still
  wrap it in ZeRO-3 so its backbone is sharded for memory, but we tell
  DeepSpeed that only the MLP head is trainable by passing
  ``model_parameters=[only-the-head]``.

* Every rank participates in training.  The rollout buffer is replicated on
  every rank (each iteration collects rollouts data-parallel then all-gathers
  them), and every rank walks only **its** stride of the mini-batch list.
  DeepSpeed's built-in gradient all-reduce combines the per-rank gradients
  into the correct global update.

* Rollouts / generation require *full* parameters, which ZeRO-3 normally
  partitions.  The launcher wraps the rollout phase in
  ``deepspeed_utils.gather_params_for_generation`` so each rank materialises
  the full weights for generation and then releases them before training
  resumes.

The clipped PPO math is identical to ``ppo_trainer.py``; only the backward /
optimizer step mechanics differ (engine.backward / engine.step instead of
loss.backward / optimizer.step).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import deepspeed
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.deepspeed_utils import (
    barrier,
    current_cuda_device,
    get_local_rank,
    get_rank,
    get_world_size,
    is_main_process,
    select_rank_shard,
)
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


def _materialise_ds_config(
    raw_config: Dict[str, Any],
    *,
    global_batch_size: int,
    world_size: int,
    learning_rate: float,
    grad_clip: float,
) -> Dict[str, Any]:
    """
    Resolve the "auto" slots in a DeepSpeed config against a concrete PPO
    batch size, learning rate, and world size.
    """
    cfg = copy.deepcopy(raw_config)

    world_size = max(1, int(world_size))
    micro_batch = max(1, int(global_batch_size) // world_size)
    effective_global = micro_batch * world_size

    cfg["train_batch_size"] = int(effective_global)
    cfg["train_micro_batch_size_per_gpu"] = int(micro_batch)
    cfg["gradient_accumulation_steps"] = 1

    if "optimizer" in cfg and isinstance(cfg["optimizer"].get("params"), dict):
        cfg["optimizer"]["params"]["lr"] = float(learning_rate)
    if "scheduler" in cfg and isinstance(cfg["scheduler"].get("params"), dict):
        cfg["scheduler"]["params"]["warmup_max_lr"] = float(learning_rate)

    cfg["gradient_clipping"] = float(grad_clip)

    return cfg


def _make_value_ds_config(policy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a DeepSpeed config for the critic.  We reuse the policy config,
    but turn off CPU offload for the (tiny) optimizer because the MLP head is
    cheap and we want fast critic updates.
    """
    value_cfg = copy.deepcopy(policy_cfg)
    zero = value_cfg.setdefault("zero_optimization", {})
    zero["offload_optimizer"] = {"device": "none"}
    return value_cfg


def _strip_accelerate_hooks(module: torch.nn.Module) -> None:
    """
    Remove ``accelerate``'s pre/post-forward hooks if present.

    HuggingFace's ``from_pretrained`` installs these when ``device_map`` is
    used.  They actively fight DeepSpeed ZeRO-3: every forward call they try
    to migrate tensors to the accelerate-tracked device, which collides with
    ZeRO-3's gathered-parameter placement and causes::

        RuntimeError: Expected all tensors to be on the same device,
        but found at least two devices, cuda:0 and cuda:1

    Silent no-op when accelerate isn't installed or no hooks are attached.
    """
    try:
        from accelerate.hooks import remove_hook_from_module
    except Exception:
        return
    remove_hook_from_module(module, recurse=True)


def _migrate_buffers_to_device(module: torch.nn.Module, device: torch.device) -> int:
    """
    Move all registered buffers on ``module`` (and every submodule) to
    ``device``.

    ZeRO-3 partitions *parameters* across ranks, but leaves registered buffers
    (e.g. Qwen2's rotary ``inv_freq``, HF causal masks, RoPE scaling tables)
    on whatever device they landed on during the pre-init CPU load.  On rank 1
    that buffer then silently stays on ``cuda:0`` (or ``cpu``) while our inputs
    are on ``cuda:1``, which blows up the first forward with::

        RuntimeError: Expected all tensors to be on the same device,
        but found at least two devices, cuda:0 and cuda:1

    We walk ``_buffers`` directly (rather than ``.to(device)`` on the whole
    module) so we don't accidentally touch ZeRO-3-partitioned parameters.
    Returns the number of buffers that were relocated.
    """
    moved = 0
    for submodule in module.modules():
        for name, buf in list(submodule._buffers.items()):
            if buf is None:
                continue
            if buf.device != device:
                submodule._buffers[name] = buf.to(device)
                moved += 1
    return moved


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

        # Resolve base config ------------------------------------------------
        config_path = ds_config or "configs/deepspeed_zero3_rl.json"
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_cfg = json.load(handle)

        world_size = get_world_size()
        policy_cfg = _materialise_ds_config(
            raw_cfg,
            global_batch_size=batch_size,
            world_size=world_size,
            learning_rate=learning_rate,
            grad_clip=max_grad_norm,
        )
        value_cfg = _make_value_ds_config(policy_cfg)

        self._world_size = world_size
        self._rank = get_rank()

        # Pin *this* rank to its local GPU before deepspeed.initialize so
        # that any tensors the engine materialises land on the correct
        # device.  We also reuse this device to fix up non-partitioned
        # buffers below.
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_rank())
        self.device = current_cuda_device()

        # Strip any leftover accelerate hooks before handing the model to
        # DeepSpeed; they would otherwise break every forward by trying to
        # migrate ZeRO-3-partitioned params to accelerate's chosen device.
        _strip_accelerate_hooks(policy_model)
        _strip_accelerate_hooks(value_model)

        # ------------------------------------------------------------------
        # Policy engine: every parameter is trainable (we merged the adapter
        # earlier).  ZeRO-3 shards params, grads and optimizer state.
        # ------------------------------------------------------------------
        for p in policy_model.parameters():
            p.requires_grad_(True)

        logger.info(
            "[rank %d] Initialising policy DeepSpeed engine (ZeRO-3, world_size=%d, micro_bs=%d)",
            self._rank,
            world_size,
            policy_cfg["train_micro_batch_size_per_gpu"],
        )
        self.policy_engine, self.policy_optimizer, _, _ = deepspeed.initialize(
            model=policy_model,
            model_parameters=[p for p in policy_model.parameters() if p.requires_grad],
            config=policy_cfg,
        )
        # ZeRO-3 only partitions *parameters* across ranks; registered
        # buffers stay where they were before initialise().  Without this
        # relocation, Qwen2's rotary ``inv_freq`` (registered with
        # persistent=False) collides with cuda:1 inputs on rank 1.
        moved = _migrate_buffers_to_device(self.policy_engine.module, self.device)
        if moved:
            logger.info(
                "[rank %d] Moved %d policy buffers to %s", self._rank, moved, self.device
            )

        # ------------------------------------------------------------------
        # Value engine: backbone frozen, only MLP head trains.  Pass the head
        # params as the trainable set so DeepSpeed only builds optimizer
        # state for those, but still shards the full model across ranks.
        # ------------------------------------------------------------------
        for p in value_model.backbone.parameters():
            p.requires_grad_(False)
        for p in value_model.value_head.parameters():
            p.requires_grad_(True)

        trainable_value_params = [
            p for p in value_model.parameters() if p.requires_grad
        ]
        logger.info(
            "[rank %d] Initialising value DeepSpeed engine (ZeRO-3, trainable_params=%d)",
            self._rank,
            len(trainable_value_params),
        )
        self.value_engine, self.value_optimizer, _, _ = deepspeed.initialize(
            model=value_model,
            model_parameters=trainable_value_params,
            config=value_cfg,
        )
        moved = _migrate_buffers_to_device(self.value_engine.module, self.device)
        if moved:
            logger.info(
                "[rank %d] Moved %d value buffers to %s", self._rank, moved, self.device
            )

        # Expose .policy / .value so the rest of the codebase that reaches
        # inside the trainer can still find the raw nn.Module.
        self.policy: torch.nn.Module = self.policy_engine.module
        self.value: ValueHead = self.value_engine.module  # type: ignore[assignment]

        self._last_checkpoint_meta: Dict[str, str] = {}

        # --- Diagnostics -------------------------------------------------
        # Parameter accounting.  Under ZeRO-3 every rank holds a shard, so
        # ``numel()`` on a partitioned param is the *shard* size (ds_tensor);
        # ``ds_numel`` is the unsharded global count.  We log both so you
        # can see the memory savings ZeRO-3 actually delivered.
        def _count_params(mod: torch.nn.Module) -> Dict[str, int]:
            total_global = 0
            total_local = 0
            for p in mod.parameters():
                g = int(getattr(p, "ds_numel", p.numel()))
                l = int(p.numel())
                total_global += g
                total_local += l
            return {"global": total_global, "local_shard": total_local}

        pol = _count_params(self.policy)
        val = _count_params(self.value)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            total = (
                torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            )
        else:
            allocated = reserved = total = 0.0

        logger.info(
            "[rank %d] Policy params: %.3fB global | %.3fB on this rank (shard ratio %.1fx)",
            self._rank,
            pol["global"] / 1e9,
            pol["local_shard"] / 1e9,
            pol["global"] / max(1, pol["local_shard"]),
        )
        logger.info(
            "[rank %d] Value params:  %.3fB global | %.3fB on this rank",
            self._rank, val["global"] / 1e9, val["local_shard"] / 1e9,
        )
        logger.info(
            "[rank %d] GPU after DS init: allocated %.2fGB / reserved %.2fGB / total %.2fGB (%.1f%% used)",
            self._rank, allocated, reserved, total,
            (allocated / total * 100.0) if total else 0.0,
        )
        logger.info(
            "[rank %d] DeepSpeed PPO trainer ready on %s (world_size=%d, micro_bs=%d)",
            self._rank, self.device, self._world_size,
            policy_cfg["train_micro_batch_size_per_gpu"],
        )


    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _policy_logits_at_state(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Return logits at the final non-pad position as float32.

        We gather only the last-position slice *before* upcasting so we do
        not materialise a [B, T, V] float32 tensor, which for Qwen-1.5B is
        roughly ``batch * seq * 150k * 4B`` — easily several GB per step.
        """
        outputs = self.policy_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits  # [B, T, V] in bf16
        last_idx = attention_mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        row_idx = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[row_idx, last_idx]  # [B, V] still bf16
        return last_logits.float()

    def _value_estimates(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.value_engine(
            input_ids=input_ids, attention_mask=attention_mask
        ).float()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run ``ppo_epochs`` of gradient updates over the rollout buffer.

        Each rank processes a **stride-sharded** slice of the mini-batches
        (rank ``r`` sees batches whose index ``i`` satisfies
        ``i % world_size == r``).  DeepSpeed's gradient all-reduce turns
        these per-rank micro-steps into a coherent synchronous update.
        """
        self.policy_engine.train()
        self.value_engine.train()

        stats: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        early_stopped = False
        update_steps = 0

        # Timing accumulators — reported once per train_step so you can see
        # where the PPO update time is actually going (forward vs backward
        # vs data transfer).
        time_forward = 0.0
        time_backward = 0.0
        time_optimizer = 0.0
        time_data = 0.0
        grad_norms: List[float] = []

        # DeepSpeed expects *per-rank micro-batches* of size
        # train_batch_size // world_size.  We therefore draw mini-batches
        # from the rollout buffer at the micro-batch size and stride-shard
        # them across ranks so that each rank consumes a different slice
        # of every "global" batch — true data parallelism.
        micro_batch = max(1, self.batch_size // max(self._world_size, 1))

        train_step_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            epoch_batches = list(
                rollout_buffer.get_batches(batch_size=micro_batch, shuffle=True)
            )
            # Drop the trailing batches so every rank processes exactly the
            # same number of .backward() calls.  Without this step an uneven
            # tail causes one rank to exit the loop while another is still
            # mid-backward, which ZeRO-3 treats as a lost collective.
            usable = len(epoch_batches) - (len(epoch_batches) % max(self._world_size, 1))
            epoch_batches = epoch_batches[:usable]
            my_batches = select_rank_shard(epoch_batches)

            for batch_idx, batch in enumerate(my_batches):
                t0 = time.perf_counter()
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                action_token_ids = batch["action_token_ids"].to(self.device, non_blocking=True)
                old_log_probs = batch["old_log_probs"].to(self.device, non_blocking=True).detach()
                old_values = batch["old_values"].to(self.device, non_blocking=True).detach()
                advantages = batch["advantages"].to(self.device, non_blocking=True)
                returns = batch["returns"].to(self.device, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                time_data += time.perf_counter() - t0

                # ---- Policy forward -----------------------------------
                t0 = time.perf_counter()
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
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().detach()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.clip_range).float().mean().detach()
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                time_forward += time.perf_counter() - t0

                # Early-stop has to be a *global* decision under DeepSpeed,
                # otherwise rank A breaks out of the loop while rank B is
                # still calling engine.backward() and waits forever for
                # rank A's gradient reduction.
                global_kl = approx_kl.clone()
                if self._world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        global_kl, op=torch.distributed.ReduceOp.AVG
                    )

                if global_kl.item() > 1.5 * self.target_kl:
                    if self._rank == 0:
                        logger.info(
                            "Early stop at epoch %d batch %d: global approx_kl=%.4f > 1.5*%.4f",
                            epoch, batch_idx, global_kl.item(), self.target_kl,
                        )
                    early_stopped = True
                    break

                # ---- Policy update ------------------------------------
                t0 = time.perf_counter()
                policy_objective = policy_loss - self.ent_coef * entropy
                self.policy_engine.backward(policy_objective)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                time_backward += time.perf_counter() - t0

                t0 = time.perf_counter()
                self.policy_engine.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                time_optimizer += time.perf_counter() - t0

                # DeepSpeed exposes the clipped gradient norm after step();
                # logging it every few batches helps catch exploding / vanishing
                # gradients without flooding the console.
                try:
                    gnorm = float(self.policy_engine.get_global_grad_norm() or 0.0)
                    if gnorm > 0.0:
                        grad_norms.append(gnorm)
                except Exception:
                    pass

                # ---- Value forward + update ---------------------------
                t0 = time.perf_counter()
                new_values = self._value_estimates(input_ids, attention_mask).squeeze(-1)
                vf_unclipped = (new_values - returns) ** 2
                values_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.clip_range_vf, self.clip_range_vf
                )
                vf_clipped = (values_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(vf_unclipped, vf_clipped).mean()
                value_objective = self.vf_coef * value_loss
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                time_forward += time.perf_counter() - t0

                t0 = time.perf_counter()
                self.value_engine.backward(value_objective)
                self.value_engine.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                time_backward += time.perf_counter() - t0

                update_steps += 1

                stats["policy_loss"].append(float(policy_loss.detach().item()))
                stats["value_loss"].append(float(value_loss.detach().item()))
                stats["entropy"].append(float(entropy.detach().item()))
                stats["approx_kl"].append(float(approx_kl.item()))
                stats["clip_fraction"].append(float(clip_fraction.item()))

                if self._rank == 0 and (batch_idx == 0 or (update_steps % 10 == 0)):
                    logger.info(
                        "  [epoch %d batch %d/%d] policy=%.4f value=%.4f entropy=%.4f "
                        "kl=%.4f clip=%.3f",
                        epoch, batch_idx + 1, len(my_batches),
                        policy_loss.item(), value_loss.item(), entropy.item(),
                        approx_kl.item(), clip_fraction.item(),
                    )

        # Make every rank reach this point before we return.  Otherwise a
        # faster rank can start the next iteration while a slower rank is
        # still in .backward(), which ZeRO-3 really doesn't enjoy.
        barrier()

        if update_steps == 0:
            logger.warning(
                "[rank %d] PPO train_step performed 0 optimizer updates (target_kl=%.4f).",
                self._rank,
                self.target_kl,
            )

        total_seconds = time.perf_counter() - train_step_start
        peak_mem_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        )
        mean_grad_norm = (
            sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        )

        if self._rank == 0:
            logger.info(
                "train_step summary: %d updates in %.1fs  |  "
                "data=%.1fs fwd=%.1fs bwd=%.1fs opt=%.1fs  |  "
                "peak_mem=%.2fGB  |  mean_grad_norm=%.3f  |  early_stop=%s",
                update_steps, total_seconds,
                time_data, time_forward, time_backward, time_optimizer,
                peak_mem_gb, mean_grad_norm,
                "yes" if early_stopped else "no",
            )

        metrics = {
            key: float(sum(values) / max(len(values), 1))
            for key, values in stats.items()
        }
        metrics["update_steps"] = float(update_steps)
        metrics["time_data_s"] = float(time_data)
        metrics["time_forward_s"] = float(time_forward)
        metrics["time_backward_s"] = float(time_backward)
        metrics["time_optimizer_s"] = float(time_optimizer)
        metrics["time_total_s"] = float(total_seconds)
        metrics["peak_mem_gb"] = float(peak_mem_gb)
        metrics["mean_grad_norm"] = float(mean_grad_norm)
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save sharded DeepSpeed checkpoints for both engines.

        ``path`` (e.g. ``.../iteration_001/checkpoint.pt``) is kept as the
        "marker" file so the existing CheckpointManager bookkeeping still
        works; the real weights live alongside it in ``policy_ds/`` and
        ``value_ds/`` sub-directories.
        """
        iteration_dir = os.path.dirname(path)
        if is_main_process():
            os.makedirs(iteration_dir, exist_ok=True)
        barrier()

        policy_dir = os.path.join(iteration_dir, "policy_ds")
        value_dir = os.path.join(iteration_dir, "value_ds")

        # Both engines need to participate in the save (sharded state).
        self.policy_engine.save_checkpoint(policy_dir)
        self.value_engine.save_checkpoint(value_dir)

        if is_main_process():
            self.tokenizer.save_pretrained(os.path.join(iteration_dir, "policy_tokenizer"))

            meta = {
                "format": "deepspeed_zero3_dual_engine",
                "policy_dir": policy_dir,
                "value_dir": value_dir,
            }
            torch.save(meta, path)
            self._last_checkpoint_meta = meta
            logger.info("DeepSpeed checkpoint saved under %s", iteration_dir)
        barrier()

    def load_checkpoint(self, path: str) -> None:
        """Restore sharded DeepSpeed checkpoints for both engines."""
        if os.path.exists(path):
            meta = torch.load(path, map_location="cpu")
            policy_dir = meta.get("policy_dir")
            value_dir = meta.get("value_dir")
        else:
            iteration_dir = os.path.dirname(path)
            policy_dir = os.path.join(iteration_dir, "policy_ds")
            value_dir = os.path.join(iteration_dir, "value_ds")

        self.policy_engine.load_checkpoint(policy_dir)
        if value_dir and os.path.isdir(value_dir):
            self.value_engine.load_checkpoint(value_dir)

        logger.info("DeepSpeed checkpoint loaded from %s", os.path.dirname(path))
