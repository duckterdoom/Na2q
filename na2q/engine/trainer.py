"""
NA²Q Trainer - Main Training Engine.

Handles training loop, logging, evaluation, and checkpointing.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict
import glob

from na2q.utils import EpisodeReplayBuffer, Logger, MetricsTracker, setup_experiment, get_device
from na2q.models.agent import NA2QAgent
from na2q.engine.collector import collect_episode
from environments.environment import make_env


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """Manages the training process for NA²Q."""
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device(config.get("device"))
        
        # Directories
        self.exp_dir = setup_experiment(config.get("log_dir", "Result"), config.get("exp_name"))
        self.checkpoints_dir = os.path.join(self.exp_dir, "checkpoints")
        self.history_dir = self.checkpoints_dir
        self.media_dir = self.exp_dir
        
        # Logger
        self.logger = Logger(self.exp_dir, experiment_name="", use_tensorboard=False)
        self.tracker = MetricsTracker()
        
        # Environment
        self.scenario = config.get("scenario", 1)
        self.num_envs = config.get("num_envs", 1)
        self.seed = config.get("seed", 42)
        self.max_steps = config.get("max_steps", 100)
        self.env_kwargs = config.get("env_kwargs", {})
        self.use_parallel = False  # Parallel env removed
        
        self._setup_environments()
        self._setup_agent()
        self._setup_buffer()
        
        # State
        self.start_episode = 0
        self.training_history = {"episode_rewards": [], "coverage_rates": [], "losses": []}
        
        if config.get("resume", False):
            self._resume_training()
    
    def _setup_environments(self):
        self.env = make_env(scenario=self.scenario, max_steps=self.max_steps, 
                           seed=self.seed, **self.env_kwargs)
        self.n_agents = self.env.n_sensors
        self.obs_dim = self.env.obs_dim
        self.state_dim = self.env.state_dim
        self.n_actions = self.env.n_actions
        
        self.eval_env = make_env(scenario=self.scenario, max_steps=self.max_steps, 
                                 seed=self.seed + 1000, **self.env_kwargs)
    
    def _setup_agent(self):
        self.agent = NA2QAgent(
            n_agents=self.n_agents, obs_dim=self.obs_dim, state_dim=self.state_dim,
            n_actions=self.n_actions, lr=self.config.get("lr", 5e-4),
            gamma=self.config.get("gamma", 0.99),
            epsilon_start=self.config.get("epsilon_start", 1.0),
            epsilon_end=self.config.get("epsilon_end", 0.05),
            epsilon_decay=self.config.get("epsilon_decay", 10000),
            target_update_interval=self.config.get("target_update_interval", 200),
            device=self.device, use_amp=self.config.get("use_amp", True),
            hidden_dim=128, rnn_hidden_dim=128, attention_hidden_dim=128
        )
    
    def _setup_buffer(self):
        buffer_capacity = self.config.get("buffer_capacity", 8000)
        effective_capacity = buffer_capacity * max(1, self.num_envs // 2)
        self.buffer = EpisodeReplayBuffer(
            capacity=effective_capacity, n_agents=self.n_agents, obs_dim=self.obs_dim,
            state_dim=self.state_dim, n_actions=self.n_actions,
            max_episode_length=self.max_steps, chunk_length=self.config.get("chunk_length", 100)
        )
    
    def _resume_training(self):
        checkpoint_path = os.path.join(self.checkpoints_dir, "final_model.pt")
        if os.path.exists(checkpoint_path):
            print(f"  Resuming from: {checkpoint_path}")
            self.agent.load(checkpoint_path)
            history_path = os.path.join(self.checkpoints_dir, "training_history.npz")
            if os.path.exists(history_path):
                old_history = np.load(history_path)
                self.training_history = {
                    "episode_rewards": list(old_history["episode_rewards"]),
                    "coverage_rates": list(old_history["coverage_rates"]),
                    "losses": list(old_history["losses"])
                }
                self.start_episode = len(self.training_history["episode_rewards"])
                print(f"  Continuing from episode {self.start_episode}")
        else:
            print(f"  Warning: No checkpoint found, starting fresh")
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    
    def train(self) -> Dict:
        """Run the training loop."""
        n_episodes = self.config.get("n_episodes", 2000)
        batch_size = self.config.get("batch_size", 32)
        learning_starts = self.config.get("learning_starts", 5000)
        eval_interval = self.config.get("eval_interval", 50)
        save_interval = self.config.get("save_interval", 100)
        
        target_episodes = self.start_episode + n_episodes
        pbar = tqdm(total=target_episodes, initial=self.start_episode, desc="Training", unit="ep")
        
        total_episodes = self.start_episode
        session_episodes = 0  # Track episodes in THIS session (for curriculum reset)
        last_eval_episode = self.start_episode
        last_save_episode = self.start_episode
        best_eval_reward = -float('inf')
        current_loss = 0.0

        try:
            while total_episodes < target_episodes:
                # Collect episode
                episode, info = collect_episode(self.env, self.agent, self.max_steps)
                self._process_episode(episode, info, total_episodes)
                self._log_progress(total_episodes, current_loss)
                total_episodes += 1
                pbar.update(1)
                
                # Update epsilon based on session episode count (resets on resume)
                self.agent.set_episode_count(session_episodes)
                
                # Curriculum (based on session episodes, not total - restarts on resume)
                session_episodes += 1
                self._update_curriculum(session_episodes, n_episodes)
                
                # Progress bar
                pbar.set_postfix({
                    "R": f"{self.tracker.get_mean('reward'):.2f}",
                    "C": f"{self.tracker.get_mean('coverage'):.1%}",
                    "ε": f"{self.agent.epsilon:.3f}",
                    "L": f"{current_loss:.3f}"
                })
                
                # Training updates
                updates_per_step = self.config.get("updates_per_step", 1)
                for _ in range(updates_per_step):
                    loss = self._training_step(batch_size, learning_starts, total_episodes)
                    if loss is not None:
                        current_loss = loss
                    self.training_history["losses"].append(loss)
                
                # Evaluation
                if total_episodes >= last_eval_episode + eval_interval:
                    last_eval_episode = total_episodes
                    best_eval_reward = self._evaluate_and_save(total_episodes, best_eval_reward)
                
                # Checkpoint
                if total_episodes >= last_save_episode + save_interval:
                    last_save_episode = total_episodes
                    self._save_checkpoint(total_episodes, n_episodes)
                
                # GPU cleanup
                if self.device == "cuda" and total_episodes % 1000 == 0:
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            self._handle_interrupt()
        
        self._final_save(best_eval_reward)
        self._print_training_report(total_episodes, best_eval_reward)
        self._cleanup()
        
        return {
            "exp_dir": self.exp_dir,
            "best_model_path": os.path.join(self.checkpoints_dir, "best_model.pt"),
            "training_history": self.training_history,
            "best_eval_reward": best_eval_reward,
            "total_episodes": total_episodes
        }
    
    def _log_progress(self, total_episodes, current_loss):
        print(f"[Ep {total_episodes}] Reward: {self.tracker.get_mean('reward'):.2f} | "
              f"Coverage: {self.tracker.get_mean('coverage'):.1%} | "
              f"Eps: {self.agent.epsilon:.3f} | Loss: {current_loss:.4f}")
    
    def _update_curriculum(self, session_episodes, total_session_episodes):
        if session_episodes % 100 != 0:
            return
        # Curriculum based on session progress, not total episodes
        ramp_episodes = total_session_episodes / 2.0
        curriculum_level = min(1.0, session_episodes / ramp_episodes)
        self.env.set_curriculum_difficulty(curriculum_level)
    
    # -------------------------------------------------------------------------
    # Episode Processing
    # -------------------------------------------------------------------------
    
    def _process_episode(self, episode, info, total_episodes):
        self.buffer.add_episode(episode)
        reward = sum(episode["rewards"])
        coverage = info.get("coverage_rate", 0)
        
        self.tracker.add("reward", reward)
        self.tracker.add("coverage", coverage)
        self.training_history["episode_rewards"].append(reward)
        self.training_history["coverage_rates"].append(coverage)
        
        self.logger.log_scalars({
            "episode/reward": reward,
            "episode/coverage_rate": coverage,
        }, total_episodes)
    
    # -------------------------------------------------------------------------
    # Training Step
    # -------------------------------------------------------------------------
    
    def _training_step(self, batch_size, learning_starts, total_episodes):
        can_train = len(self.buffer) >= batch_size and total_episodes >= learning_starts
        if not can_train:
            return 0.0
        
        batch = self.buffer.sample(batch_size)
        train_info = self.agent.train_step_fn(batch)
        loss = train_info["loss"]
        
        self.logger.log_scalar("train/loss", loss, self.agent.train_step)
        self.tracker.add("loss", loss)
        return loss
    
    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    
    def _evaluate_and_save(self, episode, best_reward):
        avg_reward, avg_coverage = self._run_evaluation()
        
        self.logger.log_scalars({
            "eval/reward_mean": avg_reward,
            "eval/coverage_mean": avg_coverage
        }, episode)
        
        tqdm.write(f"\n[Eval] Episode {episode}: Reward={avg_reward:.2f}, Coverage={avg_coverage:.1%}")
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            self.agent.save(os.path.join(self.checkpoints_dir, "best_model.pt"))
            tqdm.write("  -> New best model saved!")
            
        return best_reward
    
    def _run_evaluation(self, n_episodes=20):
        rewards, coverages = [], []
        for _ in range(n_episodes):
            obs_list, _ = self.eval_env.reset()
            observations = np.stack(obs_list)
            self.agent.init_hidden(1)
            
            episode_reward = 0
            done, truncated = False, False
            prev_actions = np.zeros(self.n_agents, dtype=np.int64)
            
            while not done and not truncated:
                avail_actions = np.stack(self.eval_env.get_avail_actions())
                actions = self.agent.select_actions(observations, prev_actions, avail_actions, evaluate=True)
                next_obs, reward, done, truncated, info = self.eval_env.step(actions.tolist())
                observations = np.stack(next_obs)
                episode_reward += reward
                prev_actions = actions
            
            rewards.append(episode_reward)
            coverages.append(info.get("coverage_rate", 0))
        
        return np.mean(rewards), np.mean(coverages)
    
    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    
    def _save_checkpoint(self, episode, max_episodes):
        path = os.path.join(self.checkpoints_dir, f"checkpoint_{episode}.pt")
        self.agent.save(path)
        
        # Cleanup old checkpoints
        if max_episodes > 2000:
            files = sorted(glob.glob(os.path.join(self.checkpoints_dir, "checkpoint_*.pt")),
                          key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for f in files[:-10]:
                if "best" not in f and "final" not in f:
                    try:
                        os.remove(f)
                    except:
                        pass
    
    def _save_history(self, episode):
        np.savez(
            os.path.join(self.history_dir, f"training_history_ep{episode}.npz"),
            episode_rewards=np.array(self.training_history["episode_rewards"]),
            coverage_rates=np.array(self.training_history["coverage_rates"]),
            losses=np.array(self.training_history["losses"])
        )
    
    def _print_training_report(self, total_episodes, best_eval_reward):
        """Print detailed training report after completion."""
        rewards = self.training_history["episode_rewards"]
        coverages = self.training_history["coverage_rates"]
        losses = [l for l in self.training_history["losses"] if l > 0]
        
        print("\n")
        print("=" * 70)
        print("                    NA²Q TRAINING REPORT")
        print("=" * 70)
        print(f"  Scenario:           {self.scenario}")
        print(f"  Total Episodes:     {total_episodes:,}")
        print(f"  Parallel Envs:      {self.num_envs}")
        print("-" * 70)
        print("  REWARD STATISTICS")
        print(f"    Final (last 100): {np.mean(rewards[-100:]):.2f}")
        print(f"    Best Episode:     {np.max(rewards):.2f}")
        print(f"    Overall Mean:     {np.mean(rewards):.2f}")
        print(f"    Std Deviation:    {np.std(rewards):.2f}")
        print("-" * 70)
        print("  COVERAGE STATISTICS")
        print(f"    Final (last 100): {np.mean(coverages[-100:])*100:.1f}%")
        print(f"    Best Episode:     {np.max(coverages)*100:.1f}%")
        print(f"    Overall Mean:     {np.mean(coverages)*100:.1f}%")
        print(f"    Std Deviation:    {np.std(coverages)*100:.1f}%")
        print("-" * 70)
        print("  TRAINING LOSS")
        if len(losses) > 0:
            print(f"    Final (last 100): {np.mean(losses[-100:]):.4f}")
            print(f"    Minimum:          {np.min(losses):.4f}")
        else:
            print(f"    No training updates")
        print("-" * 70)
        print("  MODEL SAVED")
        print(f"    Best Model:       {self.checkpoints_dir}/best_model.pt")
        print(f"    Final Model:      {self.checkpoints_dir}/final_model.pt")
        print(f"    History:          {self.history_dir}/training_history.npz")
        print("=" * 70)
        print("\n")
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def _handle_interrupt(self):
        print("\nTraining interrupted!")
        self.agent.save(os.path.join(self.checkpoints_dir, "interrupted_model.pt"))
    
    def _final_save(self, best_reward):
        final_path = os.path.join(self.checkpoints_dir, "final_model.pt")
        self.agent.save(final_path)
        
        best_path = os.path.join(self.checkpoints_dir, "best_model.pt")
        if not os.path.exists(best_path):
            import shutil
            shutil.copy(final_path, best_path)
        
        np.savez(
            os.path.join(self.history_dir, "training_history.npz"),
            episode_rewards=np.array(self.training_history["episode_rewards"]),
            coverage_rates=np.array(self.training_history["coverage_rates"]),
            losses=np.array(self.training_history["losses"])
        )
        self.logger.close()
    
    def _cleanup(self):
        if getattr(self, 'env', None):
            self.env.close()
        if self.eval_env:
            self.eval_env.close()
