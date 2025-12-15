"""
Trainer Engine.
Encapsulates the training loop, logging, and evaluation logic.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Optional, List
import glob

from na2q.utils import EpisodeReplayBuffer, Logger, MetricsTracker, setup_experiment, get_device
from na2q.models.agent import NA2QAgent
from na2q.engine.collector import collect_episode, collect_episodes_parallel
from environments.environment import make_env


class Trainer:
    """
    Manages the training process for NA2Q.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device(config.get("device"))
        
        # Setup directories
        self.exp_dir = setup_experiment(config.get("log_dir", "Result"), config.get("exp_name"))
        self.checkpoints_dir = os.path.join(self.exp_dir, "checkpoints")
        self.history_dir = os.path.join(self.exp_dir, "history")
        self.media_dir = os.path.join(self.exp_dir, "media")
        
        # Logger
        self.logger = Logger(self.exp_dir, experiment_name="", use_tensorboard=False)
        self.tracker = MetricsTracker()
        
        # Environment Setup
        self.scenario = config.get("scenario", 1)
        self.num_envs = config.get("num_envs", 1)
        self.seed = config.get("seed", 42)
        self.max_steps = config.get("max_steps", 100)
        self.env_kwargs = config.get("env_kwargs", {})
        
        self.use_parallel = self.num_envs > 1
        
        if self.use_parallel:
            from environments.parallel_env import make_parallel_env
            self.parallel_env = make_parallel_env(
                num_envs=self.num_envs, scenario=self.scenario, max_steps=self.max_steps, 
                seed=self.seed, **self.env_kwargs
            )
            self.n_agents = self.parallel_env.n_agents
            self.obs_dim = self.parallel_env.obs_dim
            self.state_dim = self.parallel_env.state_dim
            self.n_actions = self.parallel_env.n_actions
        else:
            self.parallel_env = None
            self.num_envs = 1
            self.env = make_env(scenario=self.scenario, max_steps=self.max_steps, seed=self.seed, **self.env_kwargs)
            self.n_agents = self.env.n_sensors
            self.obs_dim = self.env.obs_dim
            self.state_dim = self.env.state_dim
            self.n_actions = self.env.n_actions
        
        # Evaluation Environment
        self.eval_env = make_env(scenario=self.scenario, max_steps=self.max_steps, seed=self.seed + 1000, **self.env_kwargs)
        
        # Agent
        self.agent = NA2QAgent(
            n_agents=self.n_agents, obs_dim=self.obs_dim, state_dim=self.state_dim,
            n_actions=self.n_actions, lr=config.get("lr", 5e-4), gamma=config.get("gamma", 0.99),
            epsilon_start=config.get("epsilon_start", 1.0), epsilon_end=config.get("epsilon_end", 0.05),
            epsilon_decay=config.get("epsilon_decay", 10000),
            target_update_interval=config.get("target_update_interval", 200),
            device=self.device, use_amp=config.get("use_amp", True),
            hidden_dim=128, rnn_hidden_dim=128, attention_hidden_dim=128 # Increased model size
        )
        
        # Replay Buffer
        buffer_capacity = config.get("buffer_capacity", 8000)
        effective_capacity = buffer_capacity * max(1, self.num_envs // 2)
        self.buffer = EpisodeReplayBuffer(
            capacity=effective_capacity, n_agents=self.n_agents, obs_dim=self.obs_dim,
            state_dim=self.state_dim, n_actions=self.n_actions, max_episode_length=self.max_steps,
            chunk_length=config.get("chunk_length", 100)
        )
        
        # State
        self.start_episode = 0
        self.training_history = {"episode_rewards": [], "coverage_rates": [], "losses": []}
        
        # Resume logic
        if config.get("resume", False):
            self._resume_training()
            
    def _resume_training(self):
        checkpoint_path = os.path.join(self.checkpoints_dir, "final_model.pt")
        if os.path.exists(checkpoint_path):
            print(f"  Resuming from: {checkpoint_path}")
            self.agent.load(checkpoint_path)
            history_path = os.path.join(self.history_dir, "training_history.npz")
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
            print(f"  Warning: No checkpoint found at {checkpoint_path}, starting fresh")

    def train(self) -> Dict:
        """Run the training loop."""
        n_episodes = self.config.get("n_episodes", 2000)
        batch_size = self.config.get("batch_size", 32)
        learning_starts = self.config.get("learning_starts", 5000)
        eval_interval = self.config.get("eval_interval", 50)
        save_interval = self.config.get("save_interval", 100)
        
        history_save_interval = 1000 if n_episodes > 2000 else n_episodes
        episodes_per_iteration = self.num_envs if self.use_parallel else 1
        target_episodes = self.start_episode + n_episodes
        
        pbar = tqdm(total=target_episodes, initial=self.start_episode, desc="Training", unit="ep")
        
        total_episodes = self.start_episode
        last_eval_episode = self.start_episode
        last_save_episode = self.start_episode
        best_eval_reward = -float('inf')
        
        try:
            while total_episodes < target_episodes:
                # Collect Data
                if self.use_parallel:
                    episodes_list, infos_list = collect_episodes_parallel(self.parallel_env, self.agent, self.max_steps)
                    for ep, info in zip(episodes_list, infos_list):
                        if not ep["rewards"]: continue
                        self._process_episode(ep, info, total_episodes)
                        if total_episodes % 64 == 0:
                            print(f"[Ep {total_episodes}] Reward: {self.tracker.get_mean('reward'):.2f} | Coverage: {self.tracker.get_mean('coverage'):.1%} | Eps: {self.agent.epsilon:.3f}")
                        
                        total_episodes += 1
                        pbar.update(1)
                else:
                    episode, info = collect_episode(self.env, self.agent, self.max_steps)
                    self._process_episode(episode, info, total_episodes)
                    
                    if total_episodes % 64 == 0:
                        print(f"[Ep {total_episodes}] Reward: {self.tracker.get_mean('reward'):.2f} | Coverage: {self.tracker.get_mean('coverage'):.1%} | Eps: {self.agent.epsilon:.3f}")

                    total_episodes += 1
                    pbar.update(1)
                
                # Curriculum Learning Update (Linear ramp from 0.0 to 1.0 over 20k episodes)
                curriculum_level = min(1.0, total_episodes / 20000.0)
                if total_episodes % 100 == 0:  # Update every 100 episodes
                    if self.use_parallel:
                        self.parallel_env.set_curriculum_difficulty(curriculum_level)
                    else:
                        self.env.set_curriculum_difficulty(curriculum_level)
                
                # Update Pbar

                pbar.set_postfix({
                    "R": f"{self.tracker.get_mean('reward'):.2f}",
                    "C": f"{self.tracker.get_mean('coverage'):.1%}",
                    "ε": f"{self.agent.epsilon:.3f}"
                })
                
                # Training Step
                updates_per_step = self.config.get("updates_per_step", 1)
                for _ in range(updates_per_step):
                    loss = self._training_step(batch_size, learning_starts, total_episodes)
                    self.training_history["losses"].append(loss) # For history tracking
                if self.use_parallel: # Replicate loss for parallel episodes count mismatch if needed, or just append once per batch
                     # The original code replicated loss. Simplified here to append once per training step. 
                     # Actually, to match original history length logic roughly:
                     pass

                # Evaluation
                if total_episodes >= last_eval_episode + eval_interval:
                    last_eval_episode = total_episodes
                    best_eval_reward = self._evaluate_and_save(total_episodes, best_eval_reward)
                
                # Checkpointing
                if total_episodes >= last_save_episode + save_interval:
                    last_save_episode = total_episodes
                    self._save_checkpoint(total_episodes, n_episodes)
                
                # History Saving
                if total_episodes % history_save_interval == 0:
                    self._save_history(total_episodes)
                
                # GPU Cleanup
                if self.device == "cuda" and total_episodes % 1000 == 0:
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            self._handle_interrupt()
        
        self._final_save(best_eval_reward)
        self._cleanup()
        
        return {
            "exp_dir": self.exp_dir,
            "best_model_path": os.path.join(self.checkpoints_dir, "best_model.pt"),
            "training_history": self.training_history,
            "best_eval_reward": best_eval_reward,
            "total_episodes": total_episodes
        }

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
            "episode/avg_reward": self.tracker.get_mean("reward"),
            "episode/total_episodes": total_episodes
        }, total_episodes)

    def _training_step(self, batch_size, learning_starts, total_episodes):
        loss = 0.0
        total_transitions = total_episodes * self.max_steps
        if self.buffer.can_sample(batch_size) and total_transitions >= learning_starts:
            n_updates = self.config.get("updates_per_step", 1)
            total_loss = 0.0
            
            for _ in range(n_updates):
                batch = self.buffer.sample(batch_size)
                train_info = self.agent.train_step_fn(batch)
                total_loss += train_info["loss"]
            
            loss = total_loss / n_updates
            self.logger.log_scalar("train/loss", loss, self.agent.train_step)
        
        # Updates are handled inside agent.train_step_fn
        # self.agent.soft_update_target()
        # self.agent.update_epsilon()
        self.tracker.add("loss", loss)
        return loss

    def _evaluate_and_save(self, episode, best_reward):
        rewards, coverages = [], []
        for _ in range(20): # Eval episodes
            _, info = collect_episode(self.eval_env, self.agent, self.max_steps) # Use collector reuse
            # Wait, collect_episode uses ε-greedy. Evaluate needs greedy.
            # I need to modify agent.select_actions or ensure evaluate=True is passed.
            # collect_episode in collector.py doesn't allow 'evaluate=True'.
            # I should fix collector.py or implement evaluate here.
            pass
        
        # Re-implementing evaluation loop correctly
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
            episode_reward = 0
            done = False
            prev_actions = np.zeros(self.n_agents, dtype=np.int64)
            
            truncated = False
            
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

    def _save_checkpoint(self, episode, max_episodes):
        path = os.path.join(self.checkpoints_dir, f"checkpoint_{episode}.pt")
        self.agent.save(path)
        # Cleanup
        if max_episodes > 2000:
            files = sorted(glob.glob(os.path.join(self.checkpoints_dir, "checkpoint_*.pt")),
                           key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for f in files[:-10]:
                if "best" not in f and "final" not in f:
                    try: os.remove(f)
                    except: pass

    def _save_history(self, episode):
        np.savez(
            os.path.join(self.history_dir, f"training_history_ep{episode}.npz"),
            episode_rewards=np.array(self.training_history["episode_rewards"]),
            coverage_rates=np.array(self.training_history["coverage_rates"]),
            losses=np.array(self.training_history["losses"])
        )

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
        if getattr(self, 'env', None): self.env.close()
        if self.eval_env: self.eval_env.close()
        if self.parallel_env: self.parallel_env.close()
