"""Replay Buffer implementations for NA²Q training."""

import numpy as np
from typing import Dict, Optional
from collections import deque


class EpisodeReplayBuffer:
    """Episode-based replay buffer for recurrent agents."""
    
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int,
                 n_actions: int, max_episode_length: int = 100, chunk_length: int = 10):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.max_episode_length = max_episode_length
        self.chunk_length = chunk_length
        
        self.episodes = deque(maxlen=capacity)
        self.current_episode = {
            "observations": [], "actions": [], "rewards": [], "states": [],
            "next_observations": [], "next_states": [], "dones": [], "avail_actions": []
        }
    
    def add(self, observations: np.ndarray, actions: np.ndarray, reward: float,
            state: np.ndarray, next_observations: np.ndarray, next_state: np.ndarray,
            done: bool, avail_actions: np.ndarray):
        self.current_episode["observations"].append(observations)
        self.current_episode["actions"].append(actions)
        self.current_episode["rewards"].append(reward)
        self.current_episode["states"].append(state)
        self.current_episode["next_observations"].append(next_observations)
        self.current_episode["next_states"].append(next_state)
        self.current_episode["dones"].append(done)
        self.current_episode["avail_actions"].append(avail_actions)
        
        if done or len(self.current_episode["observations"]) >= self.max_episode_length:
            if len(self.current_episode["observations"]) > 0:
                episode = {k: np.array(v) for k, v in self.current_episode.items()}
                self.episodes.append(episode)
            self.current_episode = {k: [] for k in self.current_episode.keys()}
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of episodes with proper padding and diversity."""
        n_available = len(self.episodes)
        if n_available == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample with replacement if needed for diversity
        replace = batch_size > n_available
        indices = np.random.choice(n_available, min(batch_size, n_available), replace=replace)
        
        batch = {k: [] for k in ["observations", "actions", "rewards", "states",
                                 "next_observations", "next_states", "dones", "avail_actions"]}
        
        for idx in indices:
            episode = self.episodes[idx]
            ep_len = len(episode["observations"])
            
            # Sample a chunk from the episode
            if ep_len <= self.chunk_length:
                start_idx = 0
                end_idx = ep_len
            else:
                start_idx = np.random.randint(0, ep_len - self.chunk_length + 1)
                end_idx = start_idx + self.chunk_length
            
            for key in batch.keys():
                chunk = episode[key][start_idx:end_idx]
                batch[key].append(chunk)
        
        # Pad sequences to same length
        max_len = max(len(b) for b in batch["observations"]) if batch["observations"] else self.chunk_length
        
        for key in batch.keys():
            padded = []
            for item in batch[key]:
                if len(item) < max_len:
                    pad_shape = (max_len - len(item),) + item.shape[1:]
                    padding = np.zeros(pad_shape, dtype=item.dtype)
                    item = np.concatenate([item, padding], axis=0)
                padded.append(item)
            batch[key] = np.array(padded)
        
        return batch
    
    def add_episode(self, episode: Dict[str, np.ndarray]):
        """Add a complete episode to the buffer."""
        # Convert episode dict to the format expected by the buffer
        if len(episode["observations"]) == 0:
            return
        
        # Add each transition in the episode
        for i in range(len(episode["observations"])):
            self.add(
                observations=episode["observations"][i],
                actions=episode["actions"][i],
                reward=episode["rewards"][i],
                state=episode["states"][i],
                next_observations=episode["next_observations"][i],
                next_state=episode["next_states"][i],
                done=episode["dones"][i],
                avail_actions=episode["avail_actions"][i]
            )
    
    def can_sample(self, batch_size: int) -> bool:
        return len(self.episodes) >= batch_size
    
    def __len__(self) -> int:
        return len(self.episodes)


class SimpleReplayBuffer:
    """Simple transition replay buffer."""
    
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int, n_actions: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        self.observations = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.avail_actions = np.zeros((capacity, n_agents, n_actions), dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def add(self, observations: np.ndarray, actions: np.ndarray, reward: float,
            state: np.ndarray, next_observations: np.ndarray, next_state: np.ndarray,
            done: bool, avail_actions: np.ndarray):
        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.rewards[self.position] = reward
        self.states[self.position] = state
        self.next_observations[self.position] = next_observations
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.avail_actions[self.position] = avail_actions
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return {
            "observations": self.observations[indices][:, np.newaxis],
            "actions": self.actions[indices][:, np.newaxis],
            "rewards": self.rewards[indices][:, np.newaxis],
            "states": self.states[indices][:, np.newaxis],
            "next_observations": self.next_observations[indices][:, np.newaxis],
            "next_states": self.next_states[indices][:, np.newaxis],
            "dones": self.dones[indices][:, np.newaxis],
            "avail_actions": self.avail_actions[indices][:, np.newaxis]
        }
    
    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size
    
    def __len__(self) -> int:
        return self.size

