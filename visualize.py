"""
Visualization utilities for NA²Q training results.

Includes:
- Training metrics plots (rewards, coverage, loss)
- Video generation of trained agents
- Knowledge/interpretability export
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import json

# Use non-interactive backend for saving figures
plt.switch_backend('Agg')

from utils import get_device


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """Smooth a curve using moving average."""
    if len(values) < window:
        return np.array(values)
    
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    
    # Pad beginning
    pad_size = len(values) - len(smoothed)
    if pad_size > 0:
        smoothed = np.concatenate([values[:pad_size], smoothed])
    
    return smoothed


def plot_training_results(exp_dir: str, window: int = 50):
    """
    Generate training result plots.
    
    Creates:
    - training_dashboard.png: Combined overview
    - coverage_ratio.png: Coverage over time
    - training_losses.png: Loss curves
    """
    history_path = os.path.join(exp_dir, "training_history.npz")
    
    if not os.path.exists(history_path):
        print(f"Warning: No training history found at {history_path}")
        return
    
    data = np.load(history_path)
    
    rewards = data["episode_rewards"] if "episode_rewards" in data else []
    coverages = data["coverage_rates"] if "coverage_rates" in data else []
    losses = data["losses"] if "losses" in data else []
    
    episodes = np.arange(1, len(rewards) + 1)
    
    # Color scheme
    colors = {
        'reward': '#2E86AB',
        'coverage': '#28A745',
        'loss': '#DC3545',
        'smooth': '#FFC107'
    }
    
    # 1. Training Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NA²Q Training Dashboard - Directional Sensor Network', fontsize=14, fontweight='bold')
    
    # Reward plot
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color=colors['reward'], label='Raw')
    ax1.plot(episodes, smooth_curve(list(rewards), window), color=colors['reward'], linewidth=2, label=f'Smoothed (w={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Coverage plot
    ax2 = axes[0, 1]
    coverage_pct = np.array(coverages) * 100
    ax2.plot(episodes, coverage_pct, alpha=0.3, color=colors['coverage'], label='Raw')
    ax2.plot(episodes, smooth_curve(list(coverage_pct), window), color=colors['coverage'], linewidth=2, label=f'Smoothed (w={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Coverage Rate (%)')
    ax2.set_title('Target Coverage Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Loss plot
    ax3 = axes[1, 0]
    if len(losses) > 0 and np.any(np.array(losses) > 0):
        ax3.plot(episodes, losses, alpha=0.3, color=colors['loss'], label='Raw')
        ax3.plot(episodes, smooth_curve(list(losses), window), color=colors['loss'], linewidth=2, label=f'Smoothed (w={window})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss (TD + VAE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Training Summary
    ────────────────────────────
    Total Episodes: {len(rewards)}
    
    Reward:
      Final (avg last 100): {np.mean(rewards[-100:]):.3f}
      Best Episode: {np.max(rewards):.3f}
      Overall Mean: {np.mean(rewards):.3f}
    
    Coverage:
      Final (avg last 100): {np.mean(coverages[-100:])*100:.1f}%
      Best Episode: {np.max(coverages)*100:.1f}%
      Overall Mean: {np.mean(coverages)*100:.1f}%
    
    Loss:
      Final (avg last 100): {np.mean(losses[-100:]):.4f}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    dashboard_path = os.path.join(exp_dir, "training_dashboard.png")
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dashboard_path}")
    
    # 2. Coverage Ratio Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(episodes, 0, coverage_pct, alpha=0.3, color=colors['coverage'])
    ax.plot(episodes, coverage_pct, alpha=0.5, color=colors['coverage'])
    ax.plot(episodes, smooth_curve(list(coverage_pct), window), color=colors['coverage'], linewidth=2)
    ax.axhline(y=np.mean(coverage_pct), color='red', linestyle='--', label=f'Mean: {np.mean(coverage_pct):.1f}%')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Coverage Rate (%)', fontsize=12)
    ax.set_title('Target Coverage Ratio Over Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    coverage_path = os.path.join(exp_dir, "coverage_ratio.png")
    plt.savefig(coverage_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {coverage_path}")
    
    # 3. Training Losses Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(losses) > 0 and np.any(np.array(losses) > 0):
        ax.plot(episodes, losses, alpha=0.3, color=colors['loss'])
        ax.plot(episodes, smooth_curve(list(losses), window), color=colors['loss'], linewidth=2, label='Total Loss')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss (TD Loss + VAE Loss)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    loss_path = os.path.join(exp_dir, "training_losses.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {loss_path}")


def generate_video(
    model_path: str,
    scenario: int = 1,
    output_path: str = "results/demo.gif",
    duration: int = 15,
    fps: int = 10,
    device: Optional[str] = None,
    seed: int = 42
):
    """
    Generate video of trained agent.
    
    Creates 15-second GIF showing:
    - Grid layout
    - Sensor positions and FoV
    - Target movements
    - Tracking status
    """
    try:
        import imageio
    except ImportError:
        print("Error: imageio required for video generation. Install with: pip install imageio")
        return
    
    from environment import make_env
    from model import NA2QAgent
    
    # Auto-detect device if not specified
    device = get_device(device)
    
    # Create environment with rgb_array rendering
    env = make_env(scenario=scenario, render_mode="rgb_array", seed=seed)
    
    # Create and load agent
    agent = NA2QAgent(
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        device=device
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random policy")
    
    # Calculate frames needed
    total_frames = duration * fps
    steps_per_episode = env.max_steps
    
    print(f"Generating {duration}s video at {fps} FPS ({total_frames} frames)")
    print(f"Scenario {scenario}: {env.grid_size}×{env.grid_size} grid, {env.n_sensors} sensors, {env.n_targets} targets")
    
    frames = []
    frame_count = 0
    
    while frame_count < total_frames:
        obs_list, info = env.reset()
        observations = np.stack(obs_list)
        agent.init_hidden(1)
        
        done, truncated = False, False
        
        while not done and not truncated and frame_count < total_frames:
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                frame_count += 1
            
            # Take action
            avail_actions = np.stack(env.get_avail_actions())
            actions = agent.select_actions(observations, avail_actions, evaluate=True)
            next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
            observations = np.stack(next_obs_list)
    
    env.close()
    
    # Save video
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    if frames:
        # Calculate frame duration for target FPS
        frame_duration = 1.0 / fps
        imageio.mimsave(output_path, frames, duration=frame_duration)
        print(f"Saved video to {output_path} ({len(frames)} frames)")
    else:
        print("Warning: No frames captured")


def save_knowledge(
    model_path: str,
    exp_dir: str,
    scenario: int = 1,
    device: Optional[str] = None
):
    """
    Save training knowledge and interpretability data.
    
    Saves:
    - Model configuration
    - Training metrics summary
    - Sample agent contributions
    """
    from environment import make_env
    from model import NA2QAgent
    import torch
    
    # Auto-detect device if not specified
    device = get_device(device)
    
    knowledge_dir = os.path.join(exp_dir, "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # Load training history
    history_path = os.path.join(exp_dir, "training_history.npz")
    knowledge = {
        "scenario": scenario,
        "model_path": model_path
    }
    
    if os.path.exists(history_path):
        data = np.load(history_path)
        knowledge["training_metrics"] = {
            "total_episodes": len(data["episode_rewards"]),
            "final_reward_mean": float(np.mean(data["episode_rewards"][-100:])),
            "final_coverage_mean": float(np.mean(data["coverage_rates"][-100:])),
            "best_reward": float(np.max(data["episode_rewards"])),
            "best_coverage": float(np.max(data["coverage_rates"]))
        }
    
    # Load model and get sample contributions
    if os.path.exists(model_path):
        env = make_env(scenario=scenario)
        agent = NA2QAgent(
            n_agents=env.n_sensors,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            n_actions=env.n_actions,
            device=device
        )
        agent.load(model_path)
        
        # Get sample contributions
        obs_list, _ = env.reset()
        observations = np.stack(obs_list)
        state = env.get_state()
        agent.init_hidden(1)
        avail_actions = np.stack(env.get_avail_actions())
        actions = agent.select_actions(observations, avail_actions, evaluate=True)
        
        contribs = agent.get_interpretable_contributions(observations, state, actions)
        
        knowledge["sample_interpretability"] = {
            "individual_contributions": contribs["individual_contribs"].tolist(),
            "pairwise_contributions": contribs["pairwise_contribs"].tolist(),
            "attention_weights": contribs["attention_weights"].tolist(),
            "q_total": float(contribs["q_total"])
        }
        
        knowledge["model_config"] = {
            "n_agents": env.n_sensors,
            "obs_dim": env.obs_dim,
            "state_dim": env.state_dim,
            "n_actions": env.n_actions,
            "hidden_dim": 64,
            "rnn_hidden_dim": 64,
            "latent_dim": 16
        }
        
        env.close()
    
    # Save knowledge as JSON
    knowledge_path = os.path.join(knowledge_dir, "training_knowledge.json")
    with open(knowledge_path, 'w') as f:
        json.dump(knowledge, f, indent=2)
    
    print(f"Saved training knowledge to {knowledge_path}")
    
    return knowledge_path
