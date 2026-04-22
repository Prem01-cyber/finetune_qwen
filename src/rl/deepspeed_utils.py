"""
Utility helpers for running PPO with DeepSpeed ZeRO-3.

This module centralises the plumbing that lets a ZeRO-3-sharded actor/critic
still generate trajectories (which requires full, non-partitioned weights) and
cooperate across ranks when collecting rollouts and running evaluation.

The helpers are intentionally tolerant of being called when DeepSpeed is *not*
active: in that case they degrade gracefully to single-process semantics so
the same training script can be launched with or without DeepSpeed.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Iterable, Iterator, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed environment helpers
# ---------------------------------------------------------------------------


def _dist_initialized() -> bool:
    """Return True iff ``torch.distributed`` has been initialised."""
    try:
        return torch.distributed.is_available() and torch.distributed.is_initialized()
    except Exception:
        return False


def get_rank() -> int:
    """Process-wide rank (0 if not distributed)."""
    if _dist_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    """Number of processes in the distributed group (1 if not distributed)."""
    if _dist_initialized():
        return int(torch.distributed.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    """Local rank on this node (for ``cuda:<local_rank>`` device selection)."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return get_rank()


def is_main_process() -> bool:
    """True on the rank that is responsible for I/O, logging, evaluation, etc."""
    return get_rank() == 0


def barrier() -> None:
    """Safe distributed barrier — no-op when not distributed."""
    if _dist_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Parameter gathering for generation / inference under ZeRO-3
# ---------------------------------------------------------------------------


def _iter_module_parameters(modules: Sequence[torch.nn.Module]) -> List[torch.nn.Parameter]:
    """Flatten parameters from an arbitrary sequence of modules."""
    collected: List[torch.nn.Parameter] = []
    for m in modules:
        if m is None:
            continue
        collected.extend(list(m.parameters()))
    return collected


@contextlib.contextmanager
def gather_params_for_generation(*modules: torch.nn.Module) -> Iterator[None]:
    """
    Temporarily gather ZeRO-3-partitioned parameters so the wrapped modules
    can be used for ``.generate()`` / normal forward passes on every rank.

    When ZeRO-3 is inactive (e.g. vanilla PyTorch training) this is a no-op.
    """
    params = _iter_module_parameters(modules)
    if not params:
        yield
        return

    # Detect ZeRO-3 via DeepSpeed's parameter status attribute
    partitioned = any(
        getattr(p, "ds_status", None) is not None for p in params
    )
    if not partitioned:
        yield
        return

    try:
        import deepspeed  # type: ignore
    except Exception:
        yield
        return

    with deepspeed.zero.GatheredParameters(params, modifier_rank=None, enabled=True):
        yield


# ---------------------------------------------------------------------------
# Picking the compute device under DeepSpeed
# ---------------------------------------------------------------------------


def current_cuda_device() -> torch.device:
    """Return the CUDA device that *this* rank should use."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Distributed object exchange (trajectories, metrics, etc.)
# ---------------------------------------------------------------------------


def all_gather_objects(obj: Any) -> List[Any]:
    """
    Gather arbitrary (picklable) Python objects from every rank.

    Returns a list of length ``world_size`` with the object contributed by each
    rank. When running single-process this just returns ``[obj]``.
    """
    if not _dist_initialized() or get_world_size() == 1:
        return [obj]
    gathered: List[Any] = [None] * get_world_size()
    torch.distributed.all_gather_object(gathered, obj)
    return gathered


def broadcast_object(obj: Any, src_rank: int = 0) -> Any:
    """
    Broadcast a Python object from ``src_rank`` to all ranks.

    No-op when running single process.
    """
    if not _dist_initialized() or get_world_size() == 1:
        return obj
    container = [obj] if get_rank() == src_rank else [None]
    torch.distributed.broadcast_object_list(container, src=src_rank)
    return container[0]


# ---------------------------------------------------------------------------
# Splitting a workload across ranks
# ---------------------------------------------------------------------------


def split_count_evenly(total: int, world_size: int) -> List[int]:
    """Round-robin split of ``total`` items across ``world_size`` workers."""
    base, remainder = divmod(int(total), max(1, int(world_size)))
    counts = [base] * world_size
    for i in range(remainder):
        counts[i] += 1
    return counts


def my_share(total: int) -> int:
    """Number of items allocated to the current rank by a round-robin split."""
    return split_count_evenly(total, get_world_size())[get_rank()]


# ---------------------------------------------------------------------------
# Tiny helpers for the training loop
# ---------------------------------------------------------------------------


def select_rank_shard(items: Iterable[Any]) -> List[Any]:
    """
    Return the slice of ``items`` that belongs to the current rank (stride
    partition). Useful for sharding pre-shuffled mini-batches across ranks
    without needing a full DistributedSampler.
    """
    rank = get_rank()
    world_size = get_world_size()
    if world_size <= 1:
        return list(items)
    return [item for idx, item in enumerate(items) if (idx % world_size) == rank]
