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
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

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
    ) -> None:
        super().__init__()

        logger.info(f"Loading ValueHead backbone from {base_model_path}")

        config = AutoConfig.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        h = hidden_size or config.hidden_size

        self.backbone = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
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

        # Use the last non-padding token's hidden state as state representation
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        # Grab the representation at the final position
        cls_hidden = last_hidden[:, -1, :].to(self.value_head[0].weight.dtype)

        values = self.value_head(cls_hidden).squeeze(-1)  # [B]
        return values
