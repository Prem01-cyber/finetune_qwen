"""
Runtime monitoring utilities for PPO curriculum training.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class TrainingMonitor:
    def __init__(self, output_dir: str, disk_warning_gb: float = 5.0) -> None:
        self.output_dir = Path(output_dir)
        self.disk_warning_gb = float(disk_warning_gb)
        self.metrics_history: List[Dict[str, float]] = []
        self.disk_usage_history: List[Dict[str, float]] = []

    def log_iteration_timing(self, iteration: int, timings: Dict[str, float]) -> Dict[str, float]:
        total_seconds = float(max(1e-6, timings.get("total_seconds", 0.0)))
        rollouts = float(max(0, timings.get("num_rollouts", 0.0)))
        generated_tokens = float(max(0, timings.get("estimated_tokens_generated", 0.0)))
        derived = {
            "iteration": float(iteration),
            "rollouts_per_second": rollouts / total_seconds,
            "tokens_per_second": generated_tokens / total_seconds if generated_tokens > 0 else 0.0,
        }
        self.metrics_history.append({**timings, **derived})
        return derived

    def check_disk_space(self) -> Dict[str, float]:
        usage = shutil.disk_usage(self.output_dir)
        total_gb = float(usage.total) / (1024.0 ** 3)
        free_gb = float(usage.free) / (1024.0 ** 3)
        used_gb = float(usage.used) / (1024.0 ** 3)
        info = {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "used_percent": (used_gb / total_gb) * 100.0 if total_gb > 0 else 0.0,
        }
        self.disk_usage_history.append(info)
        if free_gb < self.disk_warning_gb:
            logger.warning(
                "Low disk space: %.2f GB free (threshold %.2f GB).",
                free_gb,
                self.disk_warning_gb,
            )
        return info

    def log_gpu_utilization(self, gpu_ids: List[int]) -> Dict[str, float]:
        if not gpu_ids:
            return {}
        gpu_query = ",".join(str(int(g)) for g in gpu_ids)
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            gpu_query,
        ]
        try:
            raw = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            logger.debug("Failed to query GPU utilization: %s", exc)
            return {}

        stats: Dict[str, float] = {}
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            index, util, mem_used, mem_total = parts
            used_pct = (float(mem_used) / max(1.0, float(mem_total))) * 100.0
            stats[f"gpu_{index}_utilization"] = float(util)
            stats[f"gpu_{index}_memory_used_mb"] = float(mem_used)
            stats[f"gpu_{index}_memory_total_mb"] = float(mem_total)
            stats[f"gpu_{index}_memory_used_percent"] = used_pct
        return stats

