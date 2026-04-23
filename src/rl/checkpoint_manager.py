"""
Checkpoint lifecycle management for PPO curriculum training.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(
        self,
        output_dir: str,
        keep_last_n: int = 2,
        keep_every_n: int = 100,
        compress_old_logs: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.keep_last_n = max(1, int(keep_last_n))
        self.keep_every_n = max(1, int(keep_every_n))
        self.compress_old_logs = bool(compress_old_logs)

    def save_checkpoint(self, iteration: int, trainer: object) -> Path:
        iteration_dir = self.output_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = iteration_dir / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        self.cleanup_old_checkpoints(current_iteration=iteration)
        return checkpoint_path

    def cleanup_old_checkpoints(self, current_iteration: int) -> Dict[str, int]:
        deleted_checkpoints = 0
        deleted_policy_dirs = 0
        compressed_logs = 0
        for iteration_dir in self._list_iteration_dirs():
            iteration = self._parse_iteration(iteration_dir.name)
            if iteration is None:
                continue
            keep_due_to_recent = iteration > (current_iteration - self.keep_last_n)
            keep_due_to_milestone = (iteration % self.keep_every_n) == 0
            if keep_due_to_recent or keep_due_to_milestone:
                continue

            checkpoint_path = iteration_dir / "checkpoint.pt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                deleted_checkpoints += 1

            # Also remove the HF-format policy snapshot (~3 GB per
            # iteration for Qwen-1.5B).  Previously only checkpoint.pt
            # was deleted, so 50-iteration runs accumulated ~135 GB of
            # zombie policy weights that keep_last_n was supposed to
            # prevent.  The currently-live policy on disk is whichever
            # iteration still satisfies keep_due_to_recent or
            # keep_due_to_milestone — those are skipped above.
            policy_dir = iteration_dir / "policy"
            if policy_dir.is_dir():
                shutil.rmtree(policy_dir)
                deleted_policy_dirs += 1

            if self.compress_old_logs:
                compressed_logs += self._compress_log_files(iteration_dir)

        return {
            "deleted_checkpoints": deleted_checkpoints,
            "deleted_policy_dirs": deleted_policy_dirs,
            "compressed_logs": compressed_logs,
        }

    def _compress_log_files(self, iteration_dir: Path) -> int:
        compressed = 0
        for filename in ("trajectories.jsonl", "metrics.json"):
            path = iteration_dir / filename
            if not path.exists():
                continue
            gz_path = path.with_suffix(path.suffix + ".gz")
            if gz_path.exists():
                continue
            with path.open("rb") as source, gzip.open(gz_path, "wb") as target:
                target.write(source.read())
            path.unlink()
            compressed += 1
        return compressed

    def _list_iteration_dirs(self) -> List[Path]:
        if not self.output_dir.exists():
            return []
        return sorted(
            [path for path in self.output_dir.iterdir() if path.is_dir() and path.name.startswith("iteration_")],
            key=lambda p: p.name,
        )

    def _parse_iteration(self, dirname: str) -> int | None:
        try:
            return int(dirname.split("_", 1)[1])
        except Exception:
            return None

