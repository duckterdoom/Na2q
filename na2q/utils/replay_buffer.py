"""Replay Buffer implementations for NA²Q training."""

import numpy as np
from typing import Dict, Optional
from collections import deque


class EpisodeReplayBuffer:
    """Episode-based replay buffer for recurrent agents."""
    
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int,
                 n_actions: int, max_episode_length: int = 100, chunk_length: int = 50):
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
        """
        Sample batch of episodes for RNN training.
        
        IMPORTANT: For RNN-based agents, we MUST sample from the BEGINNING of episodes
        so that hidden states are properly initialized. Random chunking breaks temporal learning.
        """
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
            
            # ALWAYS start from the beginning of the episode for proper RNN training
            # This ensures hidden states are initialized correctly
            start_idx = 0
            end_idx = min(ep_len, self.chunk_length)
            
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




