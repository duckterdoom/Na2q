"""
Episode Collector - Collects experience from environments.
"""

import numpy as np
from typing import Dict, Tuple

from environments.environment import DSNEnv
from na2q.models.agent import NA2QAgent


# =============================================================================
# Single Environment Collection
# =============================================================================

def collect_episode(env: DSNEnv, agent: NA2QAgent, max_steps: int = 100) -> Tuple[Dict, Dict]:
    """
    Collect one episode of experience from a single environment.
    
    Returns:
        episode: Dictionary containing all transitions
        info: Final step info dictionary
    """
    # Reset environment
    obs_list, info = env.reset()
    observations = np.stack(obs_list)
    state = env.get_state()
    agent.init_hidden(1)
    
    # Initialize episode storage
    episode = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "states": [],
        "next_observations": [],
        "next_states": [],
        "dones": [],
        "avail_actions": []
    }
    
    done, truncated = False, False
    prev_actions = np.zeros(agent.n_agents, dtype=np.int64)
    
    # Collect transitions
    while not done and not truncated:
        avail_actions = np.stack(env.get_avail_actions())
        actions = agent.select_actions(observations, prev_actions, avail_actions)
        
        next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
        next_observations = np.stack(next_obs_list)
        next_state = env.get_state()
        
        # Store transition
        episode["observations"].append(observations)
        episode["actions"].append(actions)
        episode["rewards"].append(reward)
        episode["states"].append(state)
        episode["next_observations"].append(next_observations)
        episode["next_states"].append(next_state)
        episode["dones"].append(float(done))
        episode["avail_actions"].append(avail_actions)
        
        # Update state
        observations = next_observations
        state = next_state
        prev_actions = actions
    
    return episode, info
