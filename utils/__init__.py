from .replay_buffer import EpisodeReplayBuffer, SimpleReplayBuffer
from .logger import Logger, MetricsTracker

# Device selection utility
try:
    import torch
except ImportError:
    torch = None

def get_device(device=None):
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


def setup_experiment(base_dir: str = "results", exp_name: str = None):
    """Setup experiment directory."""
    import os
    from datetime import datetime
    
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "knowledge"), exist_ok=True)
    
    return exp_dir


__all__ = ["EpisodeReplayBuffer", "SimpleReplayBuffer", "Logger", "MetricsTracker", "get_device", "setup_experiment"]

