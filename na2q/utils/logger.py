"""Logging utilities for NA²Q training."""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """Logger for training metrics and experiment tracking."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None, 
                 use_tensorboard: bool = True, max_history_size: int = 10000,
                 write_log_file: bool = False):
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"na2q_dsn_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
        
        self.metrics_history = {}
        self.episode_metrics = []
        self.log_file = os.path.join(self.log_dir, "training.log") if write_log_file else None
        self.max_history_size = max_history_size  # Limit memory for long training
    
    def log_scalar(self, name: str, value: float, step: int):
        if self.use_tensorboard:
            self.writer.add_scalar(name, value, step)
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append((step, value))
        
        # Trim history to prevent memory accumulation for long training runs
        if len(self.metrics_history[name]) > self.max_history_size:
            self.metrics_history[name] = self.metrics_history[name][-self.max_history_size:]
    
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            self.log_scalar(full_name, value, step)
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        metrics["episode"] = episode
        self.episode_metrics.append(metrics)
        
        # Trim episode metrics to prevent memory accumulation
        if len(self.episode_metrics) > self.max_history_size:
            self.episode_metrics = self.episode_metrics[-self.max_history_size:]
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"episode/{name}", value, episode)
        
        # Format metrics for display
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in metrics.items()])
        
        # Print to console but use tqdm.write() to avoid interfering with progress bar
        # This prevents duplicate/overlapping output while still showing episode details
        try:
            import sys
            if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'write'):
                # Use tqdm.write if available, otherwise regular print
                try:
                    from tqdm import tqdm
                    tqdm.write(f"Episode {episode} | {metrics_str}")
                except:
                    print(f"Episode {episode} | {metrics_str}")
        except:
            # Fallback to regular print if tqdm not available
            print(f"Episode {episode} | {metrics_str}")
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} | Episode {episode} | {metrics_str}\n")
    
    def log_config(self, config: Dict[str, Any]):
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        if self.use_tensorboard:
            self.writer.add_text("config", json.dumps(config, indent=2), 0)
    
    def save_metrics(self):
        # Disabled - training_history.npz contains the same data
        pass
    
    def close(self):
        if self.writer is not None:
            self.writer.close()


class MetricsTracker:
    """Track and compute statistics for metrics over episodes."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
    
    def add(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def get_mean(self, name: str) -> float:
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])
    
    def get_std(self, name: str) -> float:
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        return np.std(self.metrics[name])
    
    def get_summary(self) -> Dict[str, float]:
        summary = {}
        for name in self.metrics:
            summary[f"{name}_mean"] = self.get_mean(name)
            summary[f"{name}_std"] = self.get_std(name)
        return summary
    
    def reset(self):
        self.metrics = {}





