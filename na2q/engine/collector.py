"""
Episode Collector Component.
Handles interaction with environments (single or parallel) to collect experience.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from environments.environment import DSNEnv
from na2q.models.agent import NA2QAgent


def collect_episode(env: DSNEnv, agent: NA2QAgent, max_steps: int = 100) -> Tuple[Dict, Dict]:
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
    
    prev_actions = np.zeros(agent.n_agents, dtype=np.int64)
    
    while not done and not truncated:
        avail_actions = np.stack(env.get_avail_actions())
        actions = agent.select_actions(observations, prev_actions, avail_actions)
        
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
        prev_actions = actions
    
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
    
    # Initialize previous actions (zero)
    prev_actions = np.zeros((num_envs, n_agents), dtype=np.int64)
    
    step = 0
    while not all(dones) and step < max_steps:
        # Select actions for all environments at once (batch inference)
        actions, hidden_states = agent.select_actions_batch(
            observations, prev_actions, avail_actions, hidden_states, evaluate=False
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
        # Update for next step
        observations = next_obs
        states = next_states
        avail_actions = next_avail
        prev_actions = actions
        
        # Zero out prev_actions for done environments? 
        # Actually, if we reset hidden state, we should probably reset prev_actions too for consistency.
        # But here 'actions' are what we just did. If done, it matters not.
        # But hidden states ARE masked.
        prev_actions = prev_actions * (1 - terminateds[:, None].astype(int)) * (1 - truncateds[:, None].astype(int))
        
        step += 1
    
    # Handle any environments that didn't finish
    for i in range(num_envs):
        if final_infos[i] is None:
            final_infos[i] = infos[i]
    
    return episodes, final_infos
