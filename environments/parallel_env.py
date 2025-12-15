"""
Parallel Environment for efficient GPU utilization.

Runs multiple DSN environments in parallel to:
1. Avoid CPU bottleneck during training
2. Collect multiple episodes simultaneously
3. Better utilize GPU with larger batches

Based on common practices in MARL (e.g., SMAC, MPE) and the NA²Q paper.
"""

import numpy as np
from multiprocessing import Process, Pipe
from typing import List, Tuple, Dict, Optional, Any
import cloudpickle
from environments.environment import DSNEnv, make_env


def worker(remote, parent_remote, env_fn_wrapper):
    """Worker process that runs an environment."""
    parent_remote.close()
    env = env_fn_wrapper.x()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == "step":
                try:
                    obs, reward, terminated, truncated, info = env.step(data)
                    # Stack observations
                    obs_array = np.stack(obs)
                    state = env.get_state()
                    avail_actions = np.stack(env.get_avail_actions())
                    remote.send((obs_array, state, reward, terminated, truncated, info, avail_actions))
                except Exception as e:
                    # Send error info and continue (prevents worker crash)
                    remote.send((None, None, 0.0, True, True, {"error": str(e)}, None))
                
            elif cmd == "reset":
                try:
                    obs, info = env.reset(seed=data)
                    obs_array = np.stack(obs)
                    state = env.get_state()
                    avail_actions = np.stack(env.get_avail_actions())
                    remote.send((obs_array, state, info, avail_actions))
                except Exception as e:
                    # Send error info and continue (prevents worker crash)
                    remote.send((None, None, {"error": str(e)}, None))
                
            elif cmd == "get_state":
                try:
                    remote.send(env.get_state())
                except Exception as e:
                    remote.send(None)
                
            elif cmd == "get_avail_actions":
                try:
                    avail_actions = np.stack(env.get_avail_actions())
                    remote.send(avail_actions)
                except Exception as e:
                    remote.send(None)
                
            elif cmd == "get_env_info":
                try:
                    env_info = {
                        "n_agents": env.n_sensors,
                        "n_actions": env.n_actions,
                        "obs_dim": env.obs_dim,
                        "state_dim": env.state_dim,
                        "max_steps": env.max_steps,
                        "n_sensors": env.n_sensors,
                        "n_targets": env.n_targets,
                        "grid_size": env.grid_size
                    }
                    remote.send(env_info)
                except Exception as e:
                    remote.send({"error": str(e)})
                
            elif cmd == "close":
                try:
                    env.close()
                except:
                    pass
                remote.close()
                break
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
        except EOFError:
            break
        except Exception as e:
            # Log error but continue worker process (prevents crash during long training)
            print(f"Worker error: {e}")
            continue


class CloudpickleWrapper:
    """Wrapper that uses cloudpickle to serialize the environment constructor."""
    def __init__(self, x):
        self.x = x
    
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    
    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)


class ParallelEnv:
    """
    Vectorized environment that runs multiple DSNEnv instances in parallel.
    
    This allows:
    1. Collecting multiple episodes simultaneously
    2. Better GPU utilization by batching observations
    3. Avoiding CPU bottleneck during training
    
    Usage:
        env = ParallelEnv(num_envs=8, scenario=1, seed=42)
        obs, states, infos, avail_actions = env.reset()
        next_obs, next_states, rewards, dones, truncateds, infos, avail_actions = env.step(actions)
    """
    
    def __init__(
        self,
        num_envs: int = 4,
        scenario: int = 1,
        max_steps: int = 100,
        seed: int = 42,
        **env_kwargs
    ):
        self.num_envs = num_envs
        self.scenario = scenario
        self.max_steps = max_steps
        self.waiting = False
        self.closed = False
        
        # Create environment constructors
        def make_env_fn(env_id):
            def _thunk():
                env = make_env(
                    scenario=scenario,
                    max_steps=max_steps,
                    seed=seed + env_id * 1000,
                    **env_kwargs
                )
                return env
            return _thunk
        
        env_fns = [make_env_fn(i) for i in range(num_envs)]
        
        # Create processes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.processes = []
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
                daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get environment info from first env
        self.remotes[0].send(("get_env_info", None))
        self.env_info = self.remotes[0].recv()
        
        self.n_agents = self.env_info["n_agents"]
        self.n_actions = self.env_info["n_actions"]
        self.obs_dim = self.env_info["obs_dim"]
        self.state_dim = self.env_info["state_dim"]
        self.n_sensors = self.env_info["n_sensors"]
        self.n_targets = self.env_info["n_targets"]
        self.grid_size = self.env_info["grid_size"]
    
    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
        """
        Reset all environments.
        
        Returns:
            observations: [num_envs, n_agents, obs_dim]
            states: [num_envs, state_dim]
            infos: List of info dicts
            avail_actions: [num_envs, n_agents, n_actions]
        """
        if seeds is None:
            seeds = [None] * self.num_envs
            
        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))
        
        results = [remote.recv() for remote in self.remotes]
        
        observations = np.stack([r[0] for r in results])  # [num_envs, n_agents, obs_dim]
        states = np.stack([r[1] for r in results])  # [num_envs, state_dim]
        infos = [r[2] for r in results]
        avail_actions = np.stack([r[3] for r in results])  # [num_envs, n_agents, n_actions]
        
        return observations, states, infos, avail_actions
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], np.ndarray]:
        """
        Step all environments with given actions.
        
        Args:
            actions: [num_envs, n_agents] array of actions
            
        Returns:
            observations: [num_envs, n_agents, obs_dim]
            states: [num_envs, state_dim]
            rewards: [num_envs]
            terminateds: [num_envs]
            truncateds: [num_envs]
            infos: List of info dicts
            avail_actions: [num_envs, n_agents, n_actions]
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action.tolist()))
        
        results = [remote.recv() for remote in self.remotes]
        
        observations = np.stack([r[0] for r in results])  # [num_envs, n_agents, obs_dim]
        states = np.stack([r[1] for r in results])  # [num_envs, state_dim]
        rewards = np.array([r[2] for r in results])  # [num_envs]
        terminateds = np.array([r[3] for r in results])  # [num_envs]
        truncateds = np.array([r[4] for r in results])  # [num_envs]
        infos = [r[5] for r in results]
        avail_actions = np.stack([r[6] for r in results])  # [num_envs, n_agents, n_actions]
        
        return observations, states, rewards, terminateds, truncateds, infos, avail_actions
    
    def step_async(self, actions: np.ndarray):
        """Send step commands to all envs (non-blocking)."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action.tolist()))
        self.waiting = True
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], np.ndarray]:
        """Wait for step results from all envs."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        observations = np.stack([r[0] for r in results])
        states = np.stack([r[1] for r in results])
        rewards = np.array([r[2] for r in results])
        terminateds = np.array([r[3] for r in results])
        truncateds = np.array([r[4] for r in results])
        infos = [r[5] for r in results]
        avail_actions = np.stack([r[6] for r in results])
        
        return observations, states, rewards, terminateds, truncateds, infos, avail_actions
    
    def close(self):
        """Close all environments."""
        if self.closed:
            return
        
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        
        for remote in self.remotes:
            remote.send(("close", None))
        
        for process in self.processes:
            process.join()
        
        self.closed = True
    
    def __del__(self):
        if not self.closed:
            self.close()


class DummyParallelEnv:
    """
    Non-parallel version for debugging or when multiprocessing is not desired.
    Has the same interface as ParallelEnv but runs sequentially.
    """
    
    def __init__(
        self,
        num_envs: int = 4,
        scenario: int = 1,
        max_steps: int = 100,
        seed: int = 42,
        **env_kwargs
    ):
        self.num_envs = num_envs
        self.envs = [
            make_env(scenario=scenario, max_steps=max_steps, seed=seed + i * 1000, **env_kwargs)
            for i in range(num_envs)
        ]
        
        # Get env info from first env
        env = self.envs[0]
        self.n_agents = env.n_sensors
        self.n_actions = env.n_actions
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim
        self.max_steps = max_steps
        self.n_sensors = env.n_sensors
        self.n_targets = env.n_targets
        self.grid_size = env.grid_size
    
    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
        if seeds is None:
            seeds = [None] * self.num_envs
        
        results = []
        for env, seed in zip(self.envs, seeds):
            obs, info = env.reset(seed=seed)
            obs_array = np.stack(obs)
            state = env.get_state()
            avail_actions = np.stack(env.get_avail_actions())
            results.append((obs_array, state, info, avail_actions))
        
        observations = np.stack([r[0] for r in results])
        states = np.stack([r[1] for r in results])
        infos = [r[2] for r in results]
        avail_actions = np.stack([r[3] for r in results])
        
        return observations, states, infos, avail_actions
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], np.ndarray]:
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action.tolist())
            obs_array = np.stack(obs)
            state = env.get_state()
            avail_actions = np.stack(env.get_avail_actions())
            results.append((obs_array, state, reward, terminated, truncated, info, avail_actions))
        
        observations = np.stack([r[0] for r in results])
        states = np.stack([r[1] for r in results])
        rewards = np.array([r[2] for r in results])
        terminateds = np.array([r[3] for r in results])
        truncateds = np.array([r[4] for r in results])
        infos = [r[5] for r in results]
        avail_actions = np.stack([r[6] for r in results])
        
        return observations, states, rewards, terminateds, truncateds, infos, avail_actions
    
    def close(self):
        for env in self.envs:
            env.close()


def make_parallel_env(
    num_envs: int = 4,
    scenario: int = 1,
    max_steps: int = 100,
    seed: int = 42,
    use_dummy: bool = False,
    **env_kwargs
) -> ParallelEnv:
    """
    Create parallel environments.
    
    Args:
        num_envs: Number of parallel environments
        scenario: DSN scenario (1 or 2)
        max_steps: Max steps per episode
        seed: Random seed
        use_dummy: If True, use sequential DummyParallelEnv (for debugging)
        **env_kwargs: Additional env arguments
        
    Returns:
        ParallelEnv or DummyParallelEnv instance
    """
    if use_dummy:
        return DummyParallelEnv(
            num_envs=num_envs,
            scenario=scenario,
            max_steps=max_steps,
            seed=seed,
            **env_kwargs
        )
    else:
        return ParallelEnv(
            num_envs=num_envs,
            scenario=scenario,
            max_steps=max_steps,
            seed=seed,
            **env_kwargs
        )




