"""
Utility functions for NA²Q training.

Includes:
- EpisodeReplayBuffer: Experience replay for MARL
- Logger: TensorBoard logging and metrics tracking
- get_device: Automatic CUDA/CPU device selection
"""

import numpy as np
import os
from collections import deque
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Args:
        device: Optional device string ("cuda", "cpu", or None for auto-detect)
    
    Returns:
        Device string ("cuda" or "cpu")
    """
    if device is not None:
        if device.lower() == "cuda" and torch is not None and torch.cuda.is_available():
            return "cuda"
        elif device.lower() == "cpu":
            return "cpu"
        elif device.lower() == "cuda" and (torch is None or not torch.cuda.is_available()):
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        else:
            return device.lower()
    
    # Auto-detect
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class EpisodeReplayBuffer:
    """
    Episode Replay Buffer for storing and sampling complete episodes.
    
    Stores trajectories as sequences for RNN-based Q-learning.
    Supports random episode sampling with batch_size=32 (from paper).
    """
    
    def __init__(self, capacity: int = 5000, max_seq_len: int = 100):
        self.capacity = capacity
        self.max_seq_len = max_seq_len
        self.buffer = deque(maxlen=capacity)
    
    def add_episode(self, episode: Dict[str, np.ndarray]):
        """Add complete episode to buffer."""
        self.buffer.append(episode)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of episodes."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        episodes = [self.buffer[i] for i in indices]
        return self._collate_episodes(episodes)
    
    def _collate_episodes(self, episodes: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Collate multiple episodes into a batch."""
        max_len = max(len(ep["rewards"]) for ep in episodes)
        
        batch = {
            "observations": [], "actions": [], "rewards": [], "states": [],
            "next_observations": [], "next_states": [], "dones": [], "avail_actions": []
        }
        
        for ep in episodes:
            seq_len = len(ep["rewards"])
            pad_len = max_len - seq_len
            
            n_agents = ep["observations"][0].shape[0] if len(ep["observations"]) > 0 else 1
            obs_dim = ep["observations"][0].shape[1] if len(ep["observations"]) > 0 else 1
            state_dim = ep["states"][0].shape[0] if len(ep["states"]) > 0 else 1
            n_actions = ep["avail_actions"][0].shape[1] if len(ep["avail_actions"]) > 0 else 3
            
            obs = np.array(ep["observations"])
            actions = np.array(ep["actions"])
            rewards = np.array(ep["rewards"])
            states = np.array(ep["states"])
            next_obs = np.array(ep["next_observations"])
            next_states = np.array(ep["next_states"])
            dones = np.array(ep["dones"])
            avail = np.array(ep["avail_actions"])
            
            if pad_len > 0:
                obs = np.concatenate([obs, np.zeros((pad_len, n_agents, obs_dim))], axis=0)
                actions = np.concatenate([actions, np.zeros((pad_len, n_agents))], axis=0)
                rewards = np.concatenate([rewards, np.zeros(pad_len)], axis=0)
                states = np.concatenate([states, np.zeros((pad_len, state_dim))], axis=0)
                next_obs = np.concatenate([next_obs, np.zeros((pad_len, n_agents, obs_dim))], axis=0)
                next_states = np.concatenate([next_states, np.zeros((pad_len, state_dim))], axis=0)
                dones = np.concatenate([dones, np.ones(pad_len)], axis=0)
                avail = np.concatenate([avail, np.ones((pad_len, n_agents, n_actions))], axis=0)
            
            batch["observations"].append(obs)
            batch["actions"].append(actions)
            batch["rewards"].append(rewards)
            batch["states"].append(states)
            batch["next_observations"].append(next_obs)
            batch["next_states"].append(next_states)
            batch["dones"].append(dones)
            batch["avail_actions"].append(avail)
        
        return {k: np.array(v) for k, v in batch.items()}
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


class Logger:
    """
    Logger for training metrics and TensorBoard.
    """
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics_history = {}
        self.episode_metrics = []
        
        self.use_tensorboard = use_tensorboard and SummaryWriter is not None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if tag not in self.metrics_history:
            self.metrics_history[tag] = []
        self.metrics_history[tag].append((step, value))
        
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log multiple scalars."""
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)
    
    def log_episode(self, episode_metrics: Dict[str, float]):
        """Log episode-level metrics."""
        self.episode_metrics.append(episode_metrics)
    
    def save_metrics(self, filename: str = "metrics.npz"):
        """Save all metrics to file."""
        save_dict = {}
        for tag, values in self.metrics_history.items():
            steps, vals = zip(*values) if values else ([], [])
            save_dict[f"{tag}_steps"] = np.array(steps)
            save_dict[f"{tag}_values"] = np.array(vals)
        
        if self.episode_metrics:
            for key in self.episode_metrics[0].keys():
                save_dict[f"episode_{key}"] = np.array([ep[key] for ep in self.episode_metrics])
        
        np.savez(os.path.join(self.log_dir, filename), **save_dict)
    
    def close(self):
        if self.writer:
            self.writer.close()
        self.save_metrics()


class MetricsTracker:
    """Track and compute running statistics for metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
    
    def update(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        self.metrics[name].append(value)
    
    def get(self, name: str) -> float:
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])
    
    def get_all(self) -> Dict[str, float]:
        return {name: self.get(name) for name in self.metrics}


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """Smooth a curve using moving average."""
    if len(values) < window:
        return np.array(values)
    
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    
    # Pad beginning with original values
    pad_size = len(values) - len(smoothed)
    if pad_size > 0:
        smoothed = np.concatenate([values[:pad_size], smoothed])
    
    return smoothed


def setup_experiment(base_dir: str = "results", exp_name: str = None) -> str:
    """Setup experiment directory."""
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "knowledge"), exist_ok=True)
    
    return exp_dir

