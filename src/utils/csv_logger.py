"""
CSV Logger for training metrics.

Replaces wandb logging with simple CSV files that can be viewed later.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class CSVLogger:
    """
    Logger that writes metrics to CSV files for easy viewing and analysis.
    
    Each run creates a timestamped directory with:
    - metrics.csv: Main training metrics (key metrics only)
    - detailed_metrics/: Detailed metrics per iteration (JSON)
    - config.json: Configuration parameters
    - summary.json: Final summary statistics
    """
    
    # Define which metrics to include in the main CSV (keep it concise)
    KEY_METRICS = {
        "iteration", "step", "timestamp",
        # Training metrics
        "train/policy_loss", "train/value_loss", "train/entropy", 
        "train/approx_kl", "train/clip_fraction",
        # Evaluation metrics
        "eval/accuracy", "eval/correct", "eval/total",
        # Buffer/rollout metrics
        "rollout/mean_reward", "rollout/num_trajectories", "rollout/mean_length",
        # Curriculum metrics (high-level)
        "curriculum/topic_diversity", "curriculum/avg_difficulty", 
        "curriculum/avg_novelty", "curriculum/replay_ratio",
        # Performance metrics
        "perf/rollout_time", "perf/train_time", "perf/total_time",
        "perf/tokens_per_second",
        # Consensus metrics
        "consensus/rate", "consensus/answer_diversity",
        # Disk/resource metrics
        "system/disk_free_gb", "system/gpu_util_percent",
    }
    
    def __init__(
        self,
        project: str = "training",
        run_name: Optional[str] = None,
        log_dir: str = "logs",
        config: Optional[Dict[str, Any]] = None,
        log_detailed: bool = True,
    ):
        """
        Initialize CSV logger.
        
        Args:
            project: Project name (used as subdirectory)
            run_name: Optional run name, defaults to timestamp
            log_dir: Base directory for logs
            config: Optional configuration dict to save
            log_detailed: If True, save full metrics as JSON per iteration
        """
        self.project = project
        self.run_name = run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.log_detailed = log_detailed
        
        # Create log directory
        self.log_path = Path(log_dir) / project / self.run_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        if self.log_detailed:
            self.detailed_path = self.log_path / "detailed_metrics"
            self.detailed_path.mkdir(exist_ok=True)
        
        # Initialize metrics file
        self.metrics_file = self.log_path / "metrics.csv"
        self.metrics_writer = None
        self.metrics_handle = None
        self.fieldnames: List[str] = []
        self.step_count = 0
        
        # Save config
        if config:
            config_file = self.log_path / "config.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2, default=str)
        
        print(f"CSV Logger initialized: {self.log_path}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to CSV file (only key metrics) and optionally full metrics to JSON.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/iteration number
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # Save full detailed metrics to JSON if enabled
        if self.log_detailed:
            detailed_file = self.detailed_path / f"step_{step:04d}.json"
            with open(detailed_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
        
        # Flatten nested dicts
        flat_metrics = self._flatten_dict(metrics)
        flat_metrics["step"] = step
        flat_metrics["timestamp"] = datetime.now().isoformat()
        
        # Filter to only key metrics for CSV
        csv_metrics = {k: v for k, v in flat_metrics.items() 
                       if k in self.KEY_METRICS or any(k.startswith(prefix) for prefix in ["iteration"])}
        
        # Initialize CSV writer if needed
        if self.metrics_writer is None:
            # Determine initial fieldnames from key metrics
            self.fieldnames = ["step", "timestamp"] + sorted(
                [k for k in csv_metrics.keys() if k not in ["step", "timestamp"]]
            )
            self.metrics_handle = open(self.metrics_file, "w", newline="")
            self.metrics_writer = csv.DictWriter(
                self.metrics_handle,
                fieldnames=self.fieldnames,
                extrasaction="ignore"
            )
            self.metrics_writer.writeheader()
        
        # Add any new fields that match our key metrics
        new_fields = [k for k in csv_metrics.keys() if k not in self.fieldnames]
        if new_fields:
            self._add_columns(new_fields)
        
        # Write row
        self.metrics_writer.writerow(csv_metrics)
        self.metrics_handle.flush()
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
        """
        Flatten nested dictionary using separator.
        
        Example: {"train": {"loss": 0.5}} -> {"train/loss": 0.5}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to JSON string if not a simple type
                if isinstance(v, (list, tuple)):
                    v = json.dumps(v)
                elif not isinstance(v, (str, int, float, bool, type(None))):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)
    
    def _add_columns(self, new_fields: List[str]):
        """Add new columns to existing CSV by rewriting it."""
        self.fieldnames.extend(new_fields)
        
        # Read existing data
        self.metrics_handle.close()
        existing_data = []
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
        
        # Rewrite with new fieldnames
        self.metrics_handle = open(self.metrics_file, "w", newline="")
        self.metrics_writer = csv.DictWriter(
            self.metrics_handle,
            fieldnames=self.fieldnames,
            extrasaction="ignore"
        )
        self.metrics_writer.writeheader()
        for row in existing_data:
            self.metrics_writer.writerow(row)
    
    def save_summary(self, summary: Dict[str, Any]):
        """
        Save a summary dictionary to JSON.
        
        Args:
            summary: Summary statistics or final results
        """
        summary_file = self.log_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    def save_artifact(self, name: str, data: Any):
        """
        Save arbitrary data as JSON artifact.
        
        Args:
            name: Artifact name (will be used as filename)
            data: Data to save (must be JSON serializable)
        """
        artifact_file = self.log_path / f"{name}.json"
        with open(artifact_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def finish(self):
        """Close logger and clean up resources."""
        if self.metrics_handle:
            self.metrics_handle.close()
        print(f"Logs saved to: {self.log_path}")
    
    def __del__(self):
        """Ensure file handle is closed."""
        if self.metrics_handle and not self.metrics_handle.closed:
            self.metrics_handle.close()
