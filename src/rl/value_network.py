"""
Value Network (Critic) for PPO.

ValueHead wraps a frozen copy of the base language model backbone and
appends a small MLP to regress a scalar value V(s_t) ∈ ℝ.

Design notes
------------
- The backbone is loaded once with bfloat16 to fit on GPU.
- Only the MLP head (value_head) is updated during training; the
  backbone can optionally be unfrozen for fine-grained critic learning.
- The forward pass returns a 1-D tensor of shape (batch_size,) so the
  caller can do .item() for single inputs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.utils.attn_backend import select_attn_implementation

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    Critic network V_φ(s).

    Architecture
    ------------
    backbone (LM encoder, frozen by default)
        ↓  last-token hidden state  [hidden_size]
    Linear(hidden_size, 256) + ReLU
        ↓
    Linear(256, 1)
        ↓  squeeze  →  scalar V(s)

    Args:
        base_model_path : HuggingFace model id or local checkpoint path.
        freeze_backbone : If True, backbone weights are not updated.
                         Defaults to True (only head is trained).
        hidden_size     : Override backbone hidden size (auto-detected
                          from config when None).
    """

    def __init__(
        self,
        base_model_path: str,
        freeze_backbone: bool = True,
        hidden_size: Optional[int] = None,
        model_device_map: Optional[Any] = "auto",
        max_memory: Optional[dict] = None,
    ) -> None:
        super().__init__()

        logger.info(f"Loading ValueHead backbone from {base_model_path}")

        config = AutoConfig.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        h = hidden_size or config.hidden_size

        # Always load on CPU first to avoid 90% GPU allocation
        # The caller will move to GPU if needed
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": model_device_map,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": select_attn_implementation(),
        }

        self.backbone = AutoModel.from_pretrained(
            base_model_path,
            **load_kwargs,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            logger.info("Backbone frozen; only ValueHead MLP will be trained.")

        self.value_head = nn.Sequential(
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute V(s) for a batch of states.

        Args:
            input_ids      : [batch, seq_len]
            attention_mask : [batch, seq_len] (ones if None)

        Returns:
            values : [batch] — scalar value estimate per sequence
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Last *non-pad* token (right-padded batches: last valid index per row)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        last_idx = attention_mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        b = torch.arange(last_hidden.size(0), device=last_hidden.device)
        cls_hidden = last_hidden[b, last_idx].to(self.value_head[0].weight.dtype)

        values = self.value_head(cls_hidden).squeeze(-1)  # [B]
        return values

    @torch.no_grad()
    def values_at_positions(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute V(s_t) for many states in a SINGLE backbone forward pass.

        The naive rollout loop calls ``self.value(...)`` once per generated
        token, which does one full backbone forward over the growing
        sequence each step — that's O(T^2) work for T tokens.  This helper
        lets the caller run the backbone exactly once on the full
        trajectory and then pluck hidden states at the positions that
        correspond to each state s_t.

        For a trajectory with prompt length P and T generated tokens,
        state s_t (= prompt + generated[:t], t=0..T-1) is a "last token"
        at position P + t - 1 in the full sequence, so callers pass
        ``positions = torch.arange(P - 1, P + T - 1)``.

        Args:
            input_ids:
                [1, L] full trajectory (prompt + generated).  A single
                un-padded sequence — callers that need batched different-
                length trajectories should loop over them (cheap because
                each call is O(L), not O(L^2)).
            positions:
                [N] long tensor of indices into the L-axis.  Hidden states
                at these positions will be fed through the value MLP.
            attention_mask:
                Optional [1, L] mask.  Defaults to all-ones.

        Returns:
            values: [N] scalar value estimates, one per requested position,
                on the same device as ``input_ids`` and already in float32
                (so callers can safely ``.tolist()`` them for the buffer).
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # [1, L, H]

        positions = positions.to(device=hidden.device, dtype=torch.long)
        # Clamp just in case the caller requests an out-of-range position
        # (e.g. T=0 edge cases).  clamp is a no-op for valid indices.
        positions = positions.clamp(min=0, max=hidden.size(1) - 1)

        # Gather → [N, H].  Cast to the value_head's weight dtype so
        # bf16 backbone + fp32 head works regardless of how torch
        # autocast is configured on the caller side.
        gathered = hidden[0, positions].to(self.value_head[0].weight.dtype)
        values = self.value_head(gathered).squeeze(-1).float()  # [N]
        return values
