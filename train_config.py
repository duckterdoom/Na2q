"""
Training configuration presets for NA²Q.

Defines hyperparameters for different scenarios and hardware configurations.
"""

from argparse import Namespace
from copy import deepcopy
from typing import Dict


# =============================================================================
# Training Presets
# =============================================================================

TRAINING_PRESETS: Dict[int, Dict] = {
    # -------------------------------------------------------------------------
    # Scenario 1: Small-scale (5 sensors, 6 targets)
    # -------------------------------------------------------------------------
    1: {
        "device": "cuda",
        "num_envs": 1,               # Single environment (no parallel)
        "episodes": 30000,           # Reduced from 40000
        "batch_size": 256,           # Reduced for faster updates
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 4000,       # Adjusted for 30k episodes
        "target_update": 200,
        "eval_interval": 2000,
        "eval_episodes": 20,
        "save_interval": 5000,
        "buffer_capacity": 100000,   # Reduced for less memory
        "chunk_length": 25,
        "updates_per_step": 1,       # Single update per episode
        "learning_starts": 100,      # Wait for buffer to fill
        "no_amp": False,
    },
    
    # -------------------------------------------------------------------------
    # Scenario 2: Large-scale (50 sensors, 60 targets)
    # -------------------------------------------------------------------------
    2: {
        "device": "cuda",
        "num_envs": 1,               # Single environment (no parallel)
        "episodes": 20000,
        "batch_size": 128,           # Increased for better GPU utilization
        "lr": 1.0e-4,               # Lower LR for large network
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 4000,      # Adjusted for 20k episodes
        "target_update": 200,
        "eval_interval": 1000,
        "eval_episodes": 10,
        "save_interval": 2000,
        "buffer_capacity": 50000,    # Experience diversity
        "chunk_length": 25,
        "updates_per_step": 1,      # Single update per episode
        "learning_starts": 100,     # Wait for buffer to fill
        "no_amp": False,
    },
    
    # -------------------------------------------------------------------------
    # Default fallback
    # -------------------------------------------------------------------------
    0: {
        "device": "cuda",
        "num_envs": 24,
        "episodes": 30000,
        "batch_size": 128,
        "lr": 5e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 10000,
        "target_update": 200,
        "eval_interval": 50,
        "save_interval": 200,
        "buffer_capacity": 100000,
        "updates_per_step": 1,
        "chunk_length": 25,
        "no_amp": False,
    },
}

# Backward compatibility alias
DEFAULT_STRONG_GPU_PRESETS = TRAINING_PRESETS


# =============================================================================
# Helper Functions
# =============================================================================

def get_strong_gpu_settings(scenario: int = 1) -> Dict:
    """Get training settings for a scenario."""
    preset = TRAINING_PRESETS.get(scenario, TRAINING_PRESETS[0])
    return deepcopy(preset)


def format_strong_gpu_settings(scenario: int = 1) -> str:
    """Return human-readable config string."""
    cfg = get_strong_gpu_settings(scenario)
    lines = [f"Training config (Scenario {scenario}) - from train_config.py:"]
    for key, value in cfg.items():
        lines.append(f"  {key:16}: {value}")
    return "\n".join(lines)


def apply_strong_gpu_defaults(args: Namespace, override_existing: bool = False) -> Namespace:
    """Apply preset defaults to argparse namespace."""
    cfg = get_strong_gpu_settings(getattr(args, "scenario", 1))
    for key, value in cfg.items():
        if override_existing or not getattr(args, key, None):
            setattr(args, key, value)
    return args
