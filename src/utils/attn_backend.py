"""
Attention-backend selection helper.

Picks the fastest attention implementation available at runtime, with a
safe fallback ladder:

    flash_attention_2  (package `flash-attn`)
        ↓   not installed / incompatible
    sdpa               (torch.nn.functional.scaled_dot_product_attention)
        ↓   not supported by this model
    eager              (stock HF implementation, slowest)

Flash-Attn 2 is a big deal for this codebase:

* Training (PPO backward pass):
    - Turns attention activation memory from O(T^2) to O(T) per layer.
      For B=8, T=500, H=12, 28 layers of bf16 Qwen2 that is a ~1.3 GB
      saving on the backward graph, and it scales quadratically with T.
    - Fused forward+backward is 1.5-2.5x faster than SDPA on Ampere+.

* Rollouts (KV-cached `.generate()`):
    - Each generation step does an incremental attention over the full
      KV cache.  Flash is faster here too, and its lower memory footprint
      lets us keep larger prompts cached.

The helper memoizes its answer so we only probe the `flash_attn` import
once per process.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Module-level cache.  Set to None the first time select_attn_implementation
# runs; subsequent calls return the cached string.
_SELECTED: Optional[str] = None


def select_attn_implementation(
    prefer: Optional[str] = None,
    log_once: bool = True,
) -> str:
    """
    Pick the best attention backend string for
    `AutoModel{,ForCausalLM}.from_pretrained(..., attn_implementation=...)`.

    Args:
        prefer:
            If set, try this backend first.  Useful for forcing "sdpa"
            in environments where flash-attn is installed but broken
            (rare, but we've seen it on some vast.ai images).
        log_once:
            When True, emit one INFO log line the first time we pick,
            then be silent on subsequent calls.

    Returns:
        One of: "flash_attention_2", "sdpa", "eager".
    """
    global _SELECTED

    if _SELECTED is not None:
        return _SELECTED

    candidates = []
    if prefer is not None:
        candidates.append(prefer)
    # Canonical preference order.
    for name in ("flash_attention_2", "sdpa", "eager"):
        if name not in candidates:
            candidates.append(name)

    chosen = "eager"
    for name in candidates:
        if name == "flash_attention_2":
            if _flash_attention_2_available():
                chosen = "flash_attention_2"
                break
        elif name == "sdpa":
            # SDPA ships with torch >= 2.0 and is always importable.
            # The HF model class may still reject it for non-supported
            # architectures, but every modern Llama/Qwen supports it.
            chosen = "sdpa"
            break
        elif name == "eager":
            chosen = "eager"
            break

    _SELECTED = chosen
    if log_once:
        logger.info(
            "Attention backend selected: %s%s",
            chosen,
            "" if chosen == "flash_attention_2" else
            "  (flash-attn not available — install `flash-attn` for "
            "~1.5-2.5x faster attention and O(T) memory)",
        )
    return chosen


def _flash_attention_2_available() -> bool:
    """
    Return True iff `flash_attn` is importable and its version is >=2.0.

    We don't run a functional test; HF will raise a clear error at
    model-load time if the installed build is incompatible with the
    model's head dim / dtype, and we'd rather surface that than silently
    fall back and waste hours of training at 1x speed.
    """
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    version = getattr(flash_attn, "__version__", "0.0")
    try:
        major = int(str(version).split(".", 1)[0])
    except ValueError:
        return False
    return major >= 2
