# Setting for training 


from argparse import Namespace
from copy import deepcopy
from typing import Dict


DEFAULT_STRONG_GPU_PRESETS: Dict[int, Dict] = {
    1: {
        "device": "cuda",   # Use GPU
        "num_envs": 64,     # CLOUD MODE: Use all CPU cores
        "episodes": 40000,  # 10-Hour Run (Deep Convergence)
        "batch_size": 1024, # Saturate Cloud GPU
        "lr": 3.0e-4,       # Keep high LR
        "gamma": 0.99,      # Focus on long-term rewards
        "epsilon_start": 1.0, # Start with 100% random actions (Explore)
        "epsilon_end": 0.05,  # End with 5% random actions (Exploit)
        "epsilon_decay": 5000,   # Correct decay for parallel update frequency
        "target_update": 200,   # Frequent updates for stability
        "eval_interval": 2000,  # Rare evaluation
        "eval_episodes": 20,    # Robust evaluation average
        "save_interval": 5000,  # Save less often
        "buffer_capacity": 1000000, # Maximize buffer for long run
        "chunk_length": 100,    # Standard episode length
        "updates_per_step": 32, # SYNCED: High updates to match 64 envs data rate
        "learning_starts": 100, # Start immediately
        "no_amp": False,        # Speed up training (FP16)
    },
    99: {  # SANITY CHECK / TEST CONFIG
        "device": "cuda",
        "num_envs": 64,
        "episodes": 2000,   # SHORT RUN: Verify fix in 10 minutes
        "batch_size": 1024,
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 1000,   # Decays by step 1000 (Halfway)
        "target_update": 200,
        "eval_interval": 200,
        "eval_episodes": 10,
        "save_interval": 1000,
        "buffer_capacity": 200000,
        "chunk_length": 100,
        "updates_per_step": 64,  # SYNCED: 1 update per episode (64 envs -> 64 updates)
        "learning_starts": 100,
        "no_amp": False,
    },
    2: {
        "device": "cuda",
        "num_envs": 28,     # CPU: 28/32 cores (leaves headroom)
        "episodes": 20000,
        "batch_size": 128,  # GPU: Safe for 12GB VRAM (~3GB usage)
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 5000,
        "target_update": 200,
        "eval_interval": 1000,
        "eval_episodes": 20,
        "save_interval": 5000,
        "buffer_capacity": 300, # ~4200 eps. Safe for most RAM.
        "chunk_length": 100,
        "updates_per_step": 8,
        "learning_starts": 200,
        "no_amp": False,
    },
    # Fallback preset for any other scenario IDs
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
        "chunk_length": 100,
        "no_amp": False,
    },
}


def get_strong_gpu_settings(scenario: int = 1) -> Dict:
    """Get recommended settings for fast training on a strong GPU."""
    preset = DEFAULT_STRONG_GPU_PRESETS.get(scenario, DEFAULT_STRONG_GPU_PRESETS[0])
    return deepcopy(preset)


def format_strong_gpu_settings(scenario: int = 1) -> str:
    """Return a human-friendly string of the preset for printing/logging."""
    cfg = get_strong_gpu_settings(scenario)
    lines = [
        "Training config (Scenario {}):".format(scenario),
    ]
    for key, value in cfg.items():
        lines.append(f"  {key:16}: {value}")
    return "\n".join(lines)


def apply_strong_gpu_defaults(args: Namespace, override_existing: bool = False) -> Namespace:
    """
    Apply strong GPU defaults to an argparse namespace.

    Args:
        args: argparse.Namespace from main.py
        override_existing: if True, replace any existing values; otherwise only
                           fill in when the field is falsy/None.
    """
    cfg = get_strong_gpu_settings(getattr(args, "scenario", 1))
    for key, value in cfg.items():
        if override_existing or not getattr(args, key, None):
            setattr(args, key, value)
    return args
