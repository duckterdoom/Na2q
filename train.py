"""
Training functions for NA²Q on Directional Sensor Network.

Based on paper hyperparameters (Table 3, Appendix F.3):
- Learning rate: 0.0005
- Batch size: 32
- Episodes: varies by scenario
- Epsilon: 1.0 → 0.05 over 50,000 steps
- Target update: every 200 steps
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import os

from environment import DSNEnv, make_env
from model import NA2QAgent
from utils import EpisodeReplayBuffer, Logger, MetricsTracker, get_device


def collect_episode(env: DSNEnv, agent: NA2QAgent, max_steps: int = 100) -> Dict:
    """Collect one episode of experience."""
    obs_list, info = env.reset()
    observations = np.stack(obs_list)
    state = env.get_state()
    agent.init_hidden(1)
    
    episode = {
        "observations": [], "actions": [], "rewards": [], "states": [],
        "next_observations": [], "next_states": [], "dones": [], "avail_actions": []
    }
    
    done, truncated = False, False
    
    while not done and not truncated:
        avail_actions = np.stack(env.get_avail_actions())
        actions = agent.select_actions(observations, avail_actions)
        
        next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
        next_observations = np.stack(next_obs_list)
        next_state = env.get_state()
        
        episode["observations"].append(observations)
        episode["actions"].append(actions)
        episode["rewards"].append(reward)
        episode["states"].append(state)
        episode["next_observations"].append(next_observations)
        episode["next_states"].append(next_state)
        episode["dones"].append(float(done))
        episode["avail_actions"].append(avail_actions)
        
        observations = next_observations
        state = next_state
    
    return episode, info


def train(
    scenario: int = 1,
    n_episodes: int = 2000,
    max_steps: int = 100,
    batch_size: int = 32,
    buffer_capacity: int = 5000,
    lr: float = 5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 50000,
    target_update_interval: int = 200,
    eval_interval: int = 50,
    save_interval: int = 100,
    log_dir: str = "results",
    exp_name: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
    **env_kwargs
) -> Dict:
    """
    Train NA²Q agent on DSN environment.
    
    Returns training history and best model path.
    """
    from utils import setup_experiment
    
    # Auto-detect device if not specified
    device = get_device(device)
    
    # Setup
    exp_dir = setup_experiment(log_dir, exp_name)
    logger = Logger(exp_dir)
    tracker = MetricsTracker()
    
    # Create environment
    env = make_env(scenario=scenario, max_steps=max_steps, seed=seed, **env_kwargs)
    eval_env = make_env(scenario=scenario, max_steps=max_steps, seed=seed + 1000, **env_kwargs)
    
    print(f"Training NA²Q on Scenario {scenario}")
    print(f"  Grid: {env.grid_size}×{env.grid_size}")
    print(f"  Sensors: {env.n_sensors}, Targets: {env.n_targets}")
    print(f"  Obs dim: {env.obs_dim}, State dim: {env.state_dim}")
    print(f"  Device: {device}")
    if device == "cuda":
        import torch
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"  Experiment dir: {exp_dir}")
    
    # Create agent
    agent = NA2QAgent(
        n_agents=env.n_sensors, obs_dim=env.obs_dim, state_dim=env.state_dim,
        n_actions=env.n_actions, lr=lr, gamma=gamma, epsilon_start=epsilon_start,
        epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
        target_update_interval=target_update_interval, device=device
    )
    
    # Create replay buffer
    buffer = EpisodeReplayBuffer(
        capacity=buffer_capacity,
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        max_episode_length=max_steps
    )
    
    # Training loop
    best_eval_reward = -float('inf')
    training_history = {"episode_rewards": [], "coverage_rates": [], "losses": []}
    
    # For long training runs, periodically save history to disk
    history_save_interval = min(1000, n_episodes // 10) if n_episodes > 2000 else n_episodes
    
    pbar = tqdm(range(1, n_episodes + 1), desc="Training")
    
    for episode_num in pbar:
        # Collect episode
        episode, info = collect_episode(env, agent, max_steps)
        episode_reward = sum(episode["rewards"])
        coverage_rate = info.get("coverage_rate", 0)
        
        buffer.add_episode(episode)
        
        # Train if enough samples
        loss = 0.0
        if buffer.can_sample(batch_size):
            batch = buffer.sample(batch_size)
            train_info = agent.train_step_fn(batch)
            loss = train_info["loss"]
            
            logger.log_scalars({
                "train/loss": loss,
                "train/td_loss": train_info["td_loss"],
                "train/vae_loss": train_info["vae_loss"],
                "train/epsilon": train_info["epsilon"],
                "train/q_total": train_info["q_total_mean"]
            }, agent.train_step)
        
        # Log episode metrics
        tracker.add("reward", episode_reward)
        tracker.add("coverage", coverage_rate)
        tracker.add("loss", loss)
        
        training_history["episode_rewards"].append(episode_reward)
        training_history["coverage_rates"].append(coverage_rate)
        training_history["losses"].append(loss)
        
        logger.log_scalars({
            "episode/reward": episode_reward,
            "episode/coverage_rate": coverage_rate,
            "episode/avg_reward": tracker.get_mean("reward"),
            "episode/avg_coverage": tracker.get_mean("coverage")
        }, episode_num)
        
        logger.log_episode(episode_num, {
            "reward": episode_reward,
            "coverage": coverage_rate, "loss": loss
        })
        
        # Update progress bar
        pbar.set_postfix({
            "R": f"{tracker.get_mean('reward'):.2f}",
            "C": f"{tracker.get_mean('coverage'):.1%}",
            "ε": f"{agent.epsilon:.3f}"
        })
        
        # Evaluation
        if episode_num % eval_interval == 0:
            eval_rewards, eval_coverages = evaluate(eval_env, agent, n_episodes=5)
            avg_eval_reward = np.mean(eval_rewards)
            avg_eval_coverage = np.mean(eval_coverages)
            
            logger.log_scalars({
                "eval/reward_mean": avg_eval_reward,
                "eval/reward_std": np.std(eval_rewards),
                "eval/coverage_mean": avg_eval_coverage,
                "eval/coverage_std": np.std(eval_coverages)
            }, episode_num)
            
            print(f"\n[Eval] Episode {episode_num}: Reward={avg_eval_reward:.3f}, Coverage={avg_eval_coverage:.1%}")
            
            # Save best model
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                agent.save(os.path.join(exp_dir, "checkpoints", "best_model.pt"))
                print(f"  → New best model saved!")
        
        # Save checkpoint
        if episode_num % save_interval == 0:
            checkpoint_path = os.path.join(exp_dir, "checkpoints", f"checkpoint_{episode_num}.pt")
            agent.save(checkpoint_path)
            
            # Keep only last 10 checkpoints to save disk space for long runs
            if n_episodes > 2000:
                import glob
                checkpoint_files = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "checkpoint_*.pt")), 
                                        key=lambda x: int(x.split("_")[-1].split(".")[0]))
                # Keep best_model, final_model, and last 10 checkpoints
                for old_checkpoint in checkpoint_files[:-10]:
                    if "best_model" not in old_checkpoint and "final_model" not in old_checkpoint:
                        try:
                            os.remove(old_checkpoint)
                        except:
                            pass
        
        # Periodically save training history to disk for long runs
        if episode_num % history_save_interval == 0 and episode_num > 0:
            np.savez(
                os.path.join(exp_dir, f"training_history_ep{episode_num}.npz"),
                episode_rewards=np.array(training_history["episode_rewards"]),
                coverage_rates=np.array(training_history["coverage_rates"]),
                losses=np.array(training_history["losses"])
            )
    
    # Final save
    agent.save(os.path.join(exp_dir, "checkpoints", "final_model.pt"))
    logger.save_metrics()
    logger.close()
    
    # Save final training history
    np.savez(
        os.path.join(exp_dir, "training_history.npz"),
        episode_rewards=np.array(training_history["episode_rewards"]),
        coverage_rates=np.array(training_history["coverage_rates"]),
        losses=np.array(training_history["losses"])
    )
    
    # Clean up intermediate history files if final save exists
    import glob
    intermediate_files = glob.glob(os.path.join(exp_dir, "training_history_ep*.npz"))
    for f in intermediate_files:
        try:
            os.remove(f)
        except:
            pass
    
    env.close()
    eval_env.close()
    
    return {
        "exp_dir": exp_dir,
        "best_model_path": os.path.join(exp_dir, "checkpoints", "best_model.pt"),
        "training_history": training_history,
        "best_eval_reward": best_eval_reward
    }


def evaluate(env: DSNEnv, agent: NA2QAgent, n_episodes: int = 10) -> tuple:
    """Evaluate agent on environment."""
    rewards, coverages = [], []
    
    for _ in range(n_episodes):
        obs_list, _ = env.reset()
        observations = np.stack(obs_list)
        agent.init_hidden(1)
        
        episode_reward, done, truncated = 0.0, False, False
        
        while not done and not truncated:
            avail_actions = np.stack(env.get_avail_actions())
            actions = agent.select_actions(observations, avail_actions, evaluate=True)
            next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
            observations = np.stack(next_obs_list)
            episode_reward += reward
        
        rewards.append(episode_reward)
        coverages.append(info.get("coverage_rate", 0))
    
    return rewards, coverages
