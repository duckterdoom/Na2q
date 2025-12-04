"""
Training functions for NA²Q on Directional Sensor Network.

Based on paper hyperparameters (Table 3, Appendix F.3):
- Learning rate: 0.0005
- Batch size: 32
- Episodes: varies by scenario
- Epsilon: 1.0 → 0.05 over 50,000 steps
- Target update: every 200 steps

Supports parallel environments for GPU utilization:
- Multiple environments run in parallel to avoid CPU bottleneck
- Batch action selection for efficient GPU inference
- Configurable number of parallel environments
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple
import os
import torch

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


def collect_episodes_parallel(
    parallel_env,
    agent: NA2QAgent,
    max_steps: int = 100
) -> Tuple[List[Dict], List[dict]]:
    """
    Collect multiple episodes in parallel from vectorized environments.
    
    Args:
        parallel_env: ParallelEnv instance with num_envs environments
        agent: NA2QAgent for action selection
        max_steps: Maximum steps per episode
        
    Returns:
        episodes: List of episode dictionaries
        final_infos: List of final info dictionaries
    """
    num_envs = parallel_env.num_envs
    n_agents = parallel_env.n_agents
    
    # Reset all environments
    observations, states, infos, avail_actions = parallel_env.reset()
    
    # Initialize hidden states for all environments
    hidden_states = agent.model.init_hidden(num_envs).to(agent.device)
    
    # Initialize episode storage for each environment
    episodes = [{
        "observations": [], "actions": [], "rewards": [], "states": [],
        "next_observations": [], "next_states": [], "dones": [], "avail_actions": []
    } for _ in range(num_envs)]
    
    # Track which environments are done
    dones = np.zeros(num_envs, dtype=bool)
    final_infos = [None] * num_envs
    
    step = 0
    while not all(dones) and step < max_steps:
        # Select actions for all environments at once (batch inference)
        actions, hidden_states = agent.select_actions_batch(
            observations, avail_actions, hidden_states, evaluate=False
        )
        
        # Step all environments
        next_obs, next_states, rewards, terminateds, truncateds, infos, next_avail = parallel_env.step(actions)
        
        # Store transitions for each environment
        for i in range(num_envs):
            if not dones[i]:
                episodes[i]["observations"].append(observations[i])
                episodes[i]["actions"].append(actions[i])
                episodes[i]["rewards"].append(rewards[i])
                episodes[i]["states"].append(states[i])
                episodes[i]["next_observations"].append(next_obs[i])
                episodes[i]["next_states"].append(next_states[i])
                episodes[i]["dones"].append(float(terminateds[i] or truncateds[i]))
                episodes[i]["avail_actions"].append(avail_actions[i])
                
                # Check if this environment is done
                if terminateds[i] or truncateds[i]:
                    dones[i] = True
                    final_infos[i] = infos[i]
        
        # Reset hidden states for done environments
        done_mask = torch.FloatTensor(terminateds | truncateds).to(agent.device)
        done_mask_expanded = done_mask.repeat_interleave(n_agents).view(num_envs * n_agents, 1)
        hidden_states = hidden_states * (1 - done_mask_expanded)
        
        # Update for next step
        observations = next_obs
        states = next_states
        avail_actions = next_avail
        step += 1
    
    # Handle any environments that didn't finish
    for i in range(num_envs):
        if final_infos[i] is None:
            final_infos[i] = infos[i]
    
    return episodes, final_infos


def train(
    scenario: int = 1,
    n_episodes: int = 2000,
    max_steps: int = 100,
    batch_size: int = 32,
    buffer_capacity: int = 5000,
    lr: float = 5e-4,
    gamma: float = 0.99,  # From paper Table 3
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 50000,  # From paper: 50,000 steps for epsilon decay
    target_update_interval: int = 200,
    eval_interval: int = 50,
    save_interval: int = 100,
    log_dir: str = "results",
    exp_name: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
    num_envs: int = 1,  # Number of parallel environments
    **env_kwargs
) -> Dict:
    """
    Train NA²Q agent on DSN environment.
    
    Args:
        scenario: DSN scenario (1 or 2)
        n_episodes: Total number of episodes to train
        max_steps: Maximum steps per episode
        batch_size: Training batch size
        buffer_capacity: Replay buffer capacity
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Steps for epsilon decay
        target_update_interval: Steps between target network updates
        eval_interval: Episodes between evaluations
        save_interval: Episodes between checkpoint saves
        log_dir: Directory for logs
        exp_name: Experiment name
        device: Device to use (cuda/cpu/auto)
        seed: Random seed
        num_envs: Number of parallel environments (1 = sequential, >1 = parallel)
        **env_kwargs: Additional environment arguments
    
    Returns:
        Training history and best model path.
    """
    from utils import setup_experiment
    
    # Auto-detect device if not specified
    device = get_device(device)
    
    # Setup
    exp_dir = setup_experiment(log_dir, exp_name)
    logger = Logger(exp_dir)
    tracker = MetricsTracker()
    
    # Determine if using parallel environments
    use_parallel = num_envs > 1 and device == "cuda"
    
    if use_parallel:
        # Import parallel environment
        from parallel_env import make_parallel_env
        
        # Create parallel environments
        parallel_env = make_parallel_env(
            num_envs=num_envs, scenario=scenario, max_steps=max_steps, seed=seed, **env_kwargs
        )
        
        # Use first env for info
        n_agents = parallel_env.n_agents
        obs_dim = parallel_env.obs_dim
        state_dim = parallel_env.state_dim
        n_actions = parallel_env.n_actions
        grid_size = parallel_env.grid_size
        n_sensors = parallel_env.n_sensors
        n_targets = parallel_env.n_targets
    else:
        parallel_env = None
        num_envs = 1
    
    # Create single environment (for sequential training or evaluation)
    env = make_env(scenario=scenario, max_steps=max_steps, seed=seed, **env_kwargs)
    eval_env = make_env(scenario=scenario, max_steps=max_steps, seed=seed + 1000, **env_kwargs)
    
    if not use_parallel:
        n_agents = env.n_sensors
        obs_dim = env.obs_dim
        state_dim = env.state_dim
        n_actions = env.n_actions
        grid_size = env.grid_size
        n_sensors = env.n_sensors
        n_targets = env.n_targets
    
    print(f"Training NA²Q on Scenario {scenario}")
    print(f"  Grid: {grid_size}×{grid_size}")
    print(f"  Sensors: {n_sensors}, Targets: {n_targets}")
    print(f"  Obs dim: {obs_dim}, State dim: {state_dim}")
    print(f"  Device: {device}")
    print(f"  Parallel envs: {num_envs}" + (" (GPU optimized)" if use_parallel else " (sequential)"))
    if device == "cuda":
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"  Experiment dir: {exp_dir}")
    
    # Create agent
    agent = NA2QAgent(
        n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim,
        n_actions=n_actions, lr=lr, gamma=gamma, epsilon_start=epsilon_start,
        epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
        target_update_interval=target_update_interval, device=device
    )
    
    # Create replay buffer (increase capacity for parallel envs)
    effective_buffer_capacity = buffer_capacity * max(1, num_envs // 2)
    buffer = EpisodeReplayBuffer(
        capacity=effective_buffer_capacity,
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        max_episode_length=max_steps
    )
    
    # Training loop
    best_eval_reward = -float('inf')
    training_history = {"episode_rewards": [], "coverage_rates": [], "losses": []}
    
    # For long training runs (30k episodes), periodically save history to disk
    # Saves every 1000 episodes to prevent memory issues and allow recovery
    history_save_interval = 1000 if n_episodes > 2000 else n_episodes
    
    # Adjust for parallel environments
    episodes_per_iteration = num_envs if use_parallel else 1
    n_iterations = (n_episodes + episodes_per_iteration - 1) // episodes_per_iteration
    
    pbar = tqdm(range(1, n_iterations + 1), desc="Training")
    total_episodes = 0
    last_eval_episode = 0  # Track last evaluation to handle parallel env jumps
    last_save_episode = 0  # Track last save to handle parallel env jumps
    
    for iteration in pbar:
        # Collect episode(s)
        if use_parallel:
            # Collect multiple episodes in parallel
            episodes_list, infos_list = collect_episodes_parallel(parallel_env, agent, max_steps)
            
            # Process each collected episode
            for ep_idx, (episode, info) in enumerate(zip(episodes_list, infos_list)):
                if len(episode["rewards"]) == 0:
                    continue
                    
                total_episodes += 1
                episode_reward = sum(episode["rewards"])
                coverage_rate = info.get("coverage_rate", 0)
                
                buffer.add_episode(episode)
                
                # Track metrics
                tracker.add("reward", episode_reward)
                tracker.add("coverage", coverage_rate)
                training_history["episode_rewards"].append(episode_reward)
                training_history["coverage_rates"].append(coverage_rate)
            
            # Use last episode's info for logging
            episode_reward = sum(episodes_list[-1]["rewards"]) if episodes_list[-1]["rewards"] else 0
            coverage_rate = infos_list[-1].get("coverage_rate", 0) if infos_list[-1] else 0
        else:
            # Sequential episode collection
            episode, info = collect_episode(env, agent, max_steps)
            total_episodes += 1
            episode_reward = sum(episode["rewards"])
            coverage_rate = info.get("coverage_rate", 0)
            
            buffer.add_episode(episode)
            
            # Track metrics
            tracker.add("reward", episode_reward)
            tracker.add("coverage", coverage_rate)
            training_history["episode_rewards"].append(episode_reward)
            training_history["coverage_rates"].append(coverage_rate)
        
        # Train if enough samples
        # For better learning, do multiple updates per iteration
        # More updates when using parallel envs (more data collected)
        # Increased training frequency to accumulate training steps faster for epsilon decay
        loss = 0.0
        if buffer.can_sample(batch_size):
            # Scale updates with parallel envs for better GPU utilization
            # Increased base updates: 2-4 updates per episode to accumulate training steps faster
            if total_episodes < 100:
                base_updates = 2  # More updates early to start learning faster
            elif total_episodes < 1000:
                base_updates = 3  # Peak learning phase
            else:
                base_updates = 4  # More updates for faster epsilon decay and better learning
            n_updates = base_updates * max(1, num_envs // 2) if use_parallel else base_updates
            
            total_loss = 0.0
            total_td_loss = 0.0
            total_vae_loss = 0.0
            total_q_total = 0.0
            
            for _ in range(n_updates):
                batch = buffer.sample(batch_size)
                train_info = agent.train_step_fn(batch)
                total_loss += train_info["loss"]
                total_td_loss += train_info["td_loss"]
                total_vae_loss += train_info["vae_loss"]
                total_q_total += train_info["q_total_mean"]
            
            # Average over updates
            loss = total_loss / n_updates
            logger.log_scalars({
                "train/loss": loss,
                "train/td_loss": total_td_loss / n_updates,
                "train/vae_loss": total_vae_loss / n_updates,
                "train/epsilon": train_info["epsilon"],
                "train/q_total": total_q_total / n_updates
            }, agent.train_step)
        
        # Log episode metrics
        tracker.add("loss", loss)
        
        # For parallel envs, replicate loss for each episode in the batch
        if use_parallel:
            # Add loss for each episode collected in this iteration
            n_episodes_this_iter = sum(1 for ep in episodes_list if len(ep["rewards"]) > 0)
            for _ in range(n_episodes_this_iter):
                training_history["losses"].append(loss)
        else:
            training_history["losses"].append(loss)
        
        logger.log_scalars({
            "episode/reward": episode_reward,
            "episode/coverage_rate": coverage_rate,
            "episode/avg_reward": tracker.get_mean("reward"),
            "episode/avg_coverage": tracker.get_mean("coverage"),
            "episode/total_episodes": total_episodes
        }, total_episodes)
        
        logger.log_episode(total_episodes, {
            "reward": episode_reward,
            "coverage": coverage_rate, "loss": loss
        })
        
        # Update progress bar
        pbar.set_postfix({
            "Eps": total_episodes,
            "R": f"{tracker.get_mean('reward'):.2f}",
            "C": f"{tracker.get_mean('coverage'):.1%}",
            "ε": f"{agent.epsilon:.3f}"
        })
        
        # Evaluation (based on total episodes) - handle parallel env jumps
        if total_episodes >= last_eval_episode + eval_interval:
            last_eval_episode = total_episodes
            eval_rewards, eval_coverages = evaluate(eval_env, agent, n_episodes=5)
            avg_eval_reward = np.mean(eval_rewards)
            avg_eval_coverage = np.mean(eval_coverages)
            
            logger.log_scalars({
                "eval/reward_mean": avg_eval_reward,
                "eval/reward_std": np.std(eval_rewards),
                "eval/coverage_mean": avg_eval_coverage,
                "eval/coverage_std": np.std(eval_coverages)
            }, total_episodes)
            
            print(f"\n[Eval] Episode {total_episodes}: Reward={avg_eval_reward:.3f}, Coverage={avg_eval_coverage:.1%}")
            
            # Save best model
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                agent.save(os.path.join(exp_dir, "checkpoints", "best_model.pt"))
                print(f"  → New best model saved!")
        
        # Save checkpoint - handle parallel env jumps
        if total_episodes >= last_save_episode + save_interval:
            last_save_episode = total_episodes
            checkpoint_path = os.path.join(exp_dir, "checkpoints", f"checkpoint_{total_episodes}.pt")
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
        # For 30k episodes: saves every 3000 episodes to prevent memory issues
        if total_episodes % history_save_interval == 0 and total_episodes > 0:
            try:
                np.savez(
                os.path.join(exp_dir, f"training_history_ep{total_episodes}.npz"),
                episode_rewards=np.array(training_history["episode_rewards"]),
                coverage_rates=np.array(training_history["coverage_rates"]),
                losses=np.array(training_history["losses"])
            )
            except Exception as e:
                print(f"Warning: Failed to save training history at episode {total_episodes}: {e}")
                # Continue training even if history save fails
        
        # Periodic GPU memory cleanup for long training runs (every 1000 episodes)
        # This prevents GPU OOM errors during 30k episode training
        if device == "cuda" and total_episodes % 1000 == 0 and total_episodes > 0:
            torch.cuda.empty_cache()
            # Also run garbage collection to free Python objects
            import gc
            gc.collect()
        
        # Check if we've reached the target number of episodes
        if total_episodes >= n_episodes:
            break
    
    # Final save
    final_model_path = os.path.join(exp_dir, "checkpoints", "final_model.pt")
    best_model_path = os.path.join(exp_dir, "checkpoints", "best_model.pt")
    agent.save(final_model_path)
    
    # Ensure best_model.pt exists - if no evaluation improved, use final model
    if not os.path.exists(best_model_path):
        import shutil
        shutil.copy(final_model_path, best_model_path)
        print(f"  Note: Using final model as best model (no improvement during eval)")
    
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
    
    # Close parallel environments if used
    if use_parallel and parallel_env is not None:
        parallel_env.close()
    
    return {
        "exp_dir": exp_dir,
        "best_model_path": os.path.join(exp_dir, "checkpoints", "best_model.pt"),
        "training_history": training_history,
        "best_eval_reward": best_eval_reward,
        "total_episodes": total_episodes
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
