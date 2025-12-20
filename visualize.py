"""
Visualization utilities for NA²Q training results.

Includes:
- Training metrics plots (rewards, coverage, loss)
- Video generation of trained agents
- Knowledge/interpretability export
- Trend analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import json


def analyze_trend(history_path: str = 'Result/scenario1/history/training_history.npz', chunk_size: int = 1000):
    """Analyze training trends by chunking episodes."""
    if not os.path.exists(history_path):
        print(f"File not found: {history_path}")
        return
    
    data = np.load(history_path)
    rewards = data['episode_rewards']
    coverage = data['coverage_rates']
    
    num_chunks = len(rewards) // chunk_size
    
    print(f"Total episodes: {len(rewards)}")
    print(f"Analyzing in {num_chunks} chunks of {chunk_size} episodes:")
    print(f"{'Chunk':<10} {'Avg Reward':<15} {'Avg Coverage':<15}")
    print("-" * 40)
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        r_chunk = rewards[start:end]
        c_chunk = coverage[start:end]
        print(f"{i:<10} {np.mean(r_chunk):<15.4f} {np.mean(c_chunk):<15.4f}")
    
    # Last chunk if any remainder
    if len(rewards) % chunk_size != 0:
        r_chunk = rewards[num_chunks*chunk_size:]
        c_chunk = coverage[num_chunks*chunk_size:]
        print(f"{'Last':<10} {np.mean(r_chunk):<15.4f} {np.mean(c_chunk):<15.4f}")

# Use non-interactive backend for saving figures
plt.switch_backend('Agg')

from na2q.utils import get_device


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """Smooth a curve using moving average with expanding window at start."""
    if len(values) < window:
        return np.array(values)
    
    values_arr = np.array(values)
    smoothed = np.zeros(len(values))
    
    # Use expanding window for the beginning (avoids spike)
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        smoothed[i] = np.mean(values_arr[start_idx:i+1])
    
    return smoothed


def plot_training_results(exp_dir: str, window: int = 50, history_dir: Optional[str] = None, 
                          media_dir: Optional[str] = None, scenario: int = 1):
    """
    Generate professional dark-themed training dashboard.
    
    Creates:
    - train_dashboard.png: 4-panel dark theme dashboard (Reward, Coverage, Loss, Epsilon)
    - train_coverage.png: Coverage over time
    - train_losses.png: Loss curves
    """
    history_dir = history_dir or os.path.join(exp_dir, "checkpoints")
    media_dir = media_dir or os.path.join(exp_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    
    # Find history file
    history_path = os.path.join(history_dir, "training_history.npz")
    if not os.path.exists(history_path):
        history_path = os.path.join(exp_dir, "training_history.npz")
    
    if not os.path.exists(history_path):
        print(f"Warning: No training history found at {history_path}")
        return
    
    data = np.load(history_path)
    
    rewards = data["episode_rewards"] if "episode_rewards" in data else []
    coverages = data["coverage_rates"] if "coverage_rates" in data else []
    losses = data["losses"] if "losses" in data else []
    
    if len(rewards) == 0:
        print("Warning: Empty training history")
        return
    
    episodes = np.arange(1, len(rewards) + 1)
    coverage_pct = np.array(coverages) * 100
    
    # =========================================================================
    # Light Theme Dashboard
    # =========================================================================
    
    # Light color palette
    LIGHT_BG = '#ffffff'
    PANEL_BG = '#f8f9fa'
    GRID_COLOR = '#e0e0e0'
    TEXT_COLOR = '#333333'
    TITLE_COLOR = '#1a1a1a'
    
    # Chart colors
    REWARD_COLOR = '#2563eb'      # Blue
    COVERAGE_COLOR = '#16a34a'    # Green
    LOSS_COLOR = '#dc2626'        # Red
    
    # Create figure with light background
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=LIGHT_BG)
    fig.patch.set_facecolor(LIGHT_BG)
    
    # Main title
    fig.suptitle(f'NA²Q Training Dashboard - Scenario {scenario}', 
                 fontsize=18, fontweight='bold', color=TITLE_COLOR, y=0.98)
    
    def style_axis(ax, title, xlabel, ylabel, color):
        """Apply light theme styling to an axis."""
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, fontsize=14, fontweight='bold', color=TITLE_COLOR, pad=10)
        ax.set_xlabel(xlabel, fontsize=12, color=TEXT_COLOR, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=12, color=TEXT_COLOR, labelpad=8)
        ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=10, labelcolor=TEXT_COLOR)
        ax.grid(True, alpha=0.5, color=GRID_COLOR, linestyle='-')
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(0.5)
    
    # -------------------------------------------------------------------------
    # 1. Episode Reward (Top-Left)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    smoothed_rewards = smooth_curve(list(rewards), window)
    ax1.plot(episodes, smoothed_rewards, color=REWARD_COLOR, linewidth=2.5, label=f'Smoothed (w={window})')
    style_axis(ax1, 'Episode Reward', 'Episode', 'Reward', REWARD_COLOR)
    ax1.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=GRID_COLOR, 
               labelcolor=TEXT_COLOR, fontsize=9)
    
    # -------------------------------------------------------------------------
    # 2. Coverage Rate (Top-Right)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    smoothed_coverage = smooth_curve(list(coverage_pct), window)
    ax2.plot(episodes, smoothed_coverage, color=COVERAGE_COLOR, linewidth=2.5, label=f'Smoothed (w={window})')
    style_axis(ax2, 'Coverage Rate', 'Episode', 'Coverage (%)', COVERAGE_COLOR)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    
    # -------------------------------------------------------------------------
    # 3. Training Loss (Bottom-Left)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    if len(losses) > 0:
        # Filter out zero losses (before training started)
        valid_losses = [l if l > 0 else np.nan for l in losses]
        loss_x = np.linspace(1, len(rewards), len(valid_losses))
        
        if len(losses) > window:
            # Smooth only non-zero losses
            clean_losses = [l for l in losses if l > 0]
            if len(clean_losses) > window:
                smooth_loss = smooth_curve(clean_losses, window)
                smooth_x = np.linspace(1, len(rewards), len(smooth_loss))
                ax3.plot(smooth_x, smooth_loss, color=LOSS_COLOR, linewidth=2.5, label=f'Smoothed (w={window})')
    style_axis(ax3, 'Training Loss', 'Episode', 'Loss', LOSS_COLOR)
    ax3.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    
    # -------------------------------------------------------------------------
    # 4. Training Summary (Bottom-Right)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(PANEL_BG)
    ax4.axis('off')
    
    summary_text = f"""
    Training Summary
    ─────────────────────────────
    Total Episodes: {len(rewards):,}
    
    Reward:
      Highest: {np.max(rewards):.2f}
      Lowest:  {np.min(rewards):.2f}
      Mean:    {np.mean(rewards):.2f}
    
    Coverage:
      Highest: {np.max(coverages)*100:.1f}%
      Lowest:  {np.min(coverages)*100:.1f}%
      Mean:    {np.mean(coverages)*100:.1f}%
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace', color=TEXT_COLOR,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f9ff', edgecolor=GRID_COLOR, alpha=0.8))
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    dashboard_path = os.path.join(media_dir, "train_dashboard.png")
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG, edgecolor='none')
    plt.close()
    print(f"Saved: {dashboard_path}")
    
    # =========================================================================
    # Coverage Ratio Chart (Separate) - Clean visualization
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=LIGHT_BG)
    ax.set_facecolor(PANEL_BG)
    
    # Use larger smoothing window for cleaner chart
    smooth_window = max(100, window * 2)
    smoothed_coverage = smooth_curve(list(coverage_pct), smooth_window)
    
    # Plot smoothed line only (clean look)
    ax.plot(episodes, smoothed_coverage, color=COVERAGE_COLOR, linewidth=2.5, label=f'Smoothed (w={smooth_window})')
    
    # Mean line
    ax.axhline(y=np.mean(coverage_pct), color='#f59e0b', linestyle='--', linewidth=2,
               alpha=0.8, label=f'Mean: {np.mean(coverage_pct):.1f}%')
    
    ax.set_xlabel('Episode', fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel('Coverage Rate (%)', fontsize=12, color=TEXT_COLOR)
    ax.set_title(f'NA²Q Target Coverage - Scenario {scenario}', fontsize=14, fontweight='bold', color=TITLE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.set_ylim(0, 105)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    
    coverage_path = os.path.join(media_dir, "train_coverage.png")
    plt.savefig(coverage_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close()
    print(f"Saved: {coverage_path}")
    
    # =========================================================================
    # Training Losses Chart (Separate)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=LIGHT_BG)
    ax.set_facecolor(PANEL_BG)
    
    if len(losses) > 0:
        clean_losses = [l for l in losses if l > 0]
        loss_x = np.linspace(1, len(rewards), len(clean_losses))
        ax.plot(loss_x, clean_losses, alpha=0.4, color=LOSS_COLOR, linewidth=0.5)
        
        if len(clean_losses) > window:
            smooth_loss = np.convolve(clean_losses, np.ones(window)/window, mode='valid')
            smooth_x = np.linspace(1, len(rewards), len(smooth_loss))
            ax.plot(smooth_x, smooth_loss, color=LOSS_COLOR, linewidth=2.5, label='Total Loss')
        
        # Mean line
        mean_loss = np.mean(clean_losses)
        ax.axhline(y=mean_loss, color='#f59e0b', linestyle='--', linewidth=2,
                   alpha=0.8, label=f'Mean: {mean_loss:.4f}')
    
    ax.set_xlabel('Episode', fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel('Loss', fontsize=12, color=TEXT_COLOR)
    ax.set_title(f'NA²Q Training Loss - Scenario {scenario}', fontsize=14, fontweight='bold', color=TITLE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    
    loss_path = os.path.join(media_dir, "train_losses.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close()
    print(f"Saved: {loss_path}")


def plot_test_results(results: dict, output_path: str, scenario: int = 1):
    """
    Generate test results chart.
    
    Creates:
    - test_results.png: Bar chart with coverage distribution and summary stats
    """
    episode_rewards = results.get("episode_rewards", [])
    coverage_rates = results.get("coverage_rates", [])
    
    if len(episode_rewards) == 0:
        print("Warning: No test results to plot")
        return
    
    coverage_pct = np.array(coverage_rates) * 100
    n_test_episodes = len(episode_rewards)
    
    # Try to get training episodes count from training_history
    training_episodes = "N/A"
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(script_dir, "Result", f"scenario{scenario}", "history", "training_history.npz")
        if os.path.exists(history_path):
            data = np.load(history_path)
            training_episodes = len(data.get("episode_rewards", []))
    except:
        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'NA²Q Test Results - Scenario {scenario}', fontsize=14, fontweight='bold')
    
    # Left: Coverage distribution histogram
    ax1 = axes[0]
    ax1.hist(coverage_pct, bins=min(20, n_test_episodes), color='#28A745', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(coverage_pct), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(coverage_pct):.1f}%')
    ax1.set_xlabel('Coverage Rate (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Coverage Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # Right: Summary statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    summary_text = f"""
    Test Results Summary
    ────────────────────────────────
    Model trained on: {training_episodes} episodes
    Test episodes: {n_test_episodes}
    
    Coverage Rate:
      Mean:    {np.mean(coverage_pct):6.2f}%
      Std:     {np.std(coverage_pct):6.2f}%
      Best:    {np.max(coverage_pct):6.2f}%
      Worst:   {np.min(coverage_pct):6.2f}%
    
    Episode Reward:
      Mean:    {np.mean(episode_rewards):8.3f}
      Std:     {np.std(episode_rewards):8.3f}
      Best:    {np.max(episode_rewards):8.3f}
      Worst:   {np.min(episode_rewards):8.3f}
    """
    
    ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_video(
    model_path: str,
    scenario: int = 1,
    output_path: str = "results/demo.gif",
    duration: int = 20,
    fps: int = 10,
    device: Optional[str] = None,
    seed: int = 42
):
    """
    Generate video of trained agent.
    
    Creates 20-second GIF showing:
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
    
    from environments.environment import make_env
    from na2q.models import NA2QAgent
    
    # Auto-detect device if not specified
    device = get_device(device)
    
    # Create environment with rgb_array rendering
    env = make_env(scenario=scenario, render_mode="rgb_array", seed=seed)
    
    # Create and load agent (use hidden_dim=128 to match trainer)
    agent = NA2QAgent(
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=128,
        rnn_hidden_dim=128,
        attention_hidden_dim=128,
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
        prev_actions = np.zeros(env.n_sensors, dtype=np.int64)
        
        done, truncated = False, False
        
        while not done and not truncated and frame_count < total_frames:
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                frame_count += 1
            
            # Take action
            avail_actions = np.stack(env.get_avail_actions())
            actions = agent.select_actions(observations, prev_actions, avail_actions, evaluate=True)
            next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
            observations = np.stack(next_obs_list)
            prev_actions = actions
    
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
    from environments.environment import make_env
    from na2q.models import NA2QAgent
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
