"""
Directional Sensor Network (DSN) Environment for Multi-Agent Reinforcement Learning.

Based on EXPERIMENT ENVIRONMENT specification:
- Problem: Tracking randomly moving targets in WMSN/DSN
- Dec-POMDP formulation: ⟨N, S, {Ai}, {Oi}, R, Pr, Z⟩
- Observation: oij = (i, j, ρij, αij) in polar coordinates
- Actions: TurnLeft (-5°), Stay, TurnRight (+5°)
- Goal: Maximize number of tracked targets

Scenarios:
- Scenario 1: 3×3 grid, 5 sensors (cells 1,3,5,7,9), 6 targets, ρmax=18m
- Scenario 2: 10×10 grid, 50 sensors (50% probability), 60 targets, ρmax=18m
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches


class DSNEnv(gym.Env):
    """Directional Sensor Network Environment for Target Tracking."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Action constants
    ACTION_TURN_LEFT = 0   # δi,t+1 = δi,t - 5°
    ACTION_STAY = 1        # δi,t+1 = δi,t
    ACTION_TURN_RIGHT = 2  # δi,t+1 = δi,t + 5°
    
    def __init__(
        self,
        scenario: int = 1,
        n_sensors: Optional[int] = None,
        n_targets: Optional[int] = None,
        grid_size: Optional[int] = None,
        cell_size: float = 20.0,          # 20m × 20m per cell
        sensing_range: float = 18.0,       # ρmax = 18m
        fov_angle: float = 60.0,           # αmax = 60°
        rotation_step: float = 5.0,        # ±5° per action
        max_steps: int = 100,
        target_speed_range: Tuple[float, float] = (0.5, 2.0),
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.scenario = scenario
        self.cell_size = cell_size
        self.sensing_range = sensing_range
        self.fov_angle = np.radians(fov_angle)
        self.rotation_step = np.radians(rotation_step)
        self.max_steps = max_steps
        self.target_speed_range = target_speed_range
        self.render_mode = render_mode
        
        # Configure scenario from EXPERIMENT ENVIRONMENT spec
        if scenario == 1:
            # Small-scale: 3×3 grid, 5 sensors at cells 1,3,5,7,9, 6 targets
            self.grid_size = grid_size if grid_size else 3
            self.n_sensors = n_sensors if n_sensors else 5
            self.n_targets = n_targets if n_targets else 6
        elif scenario == 2:
            # Large-scale: 10×10 grid, 50 sensors (50% probability), 60 targets
            self.grid_size = grid_size if grid_size else 10
            self.n_sensors = n_sensors if n_sensors else 50
            self.n_targets = n_targets if n_targets else 60
        else:
            self.grid_size = grid_size if grid_size else 3
            self.n_sensors = n_sensors if n_sensors else 5
            self.n_targets = n_targets if n_targets else 6
        
        self.field_size = self.grid_size * self.cell_size
        self.np_random = np.random.default_rng(seed)
        
        # Action space: 3 discrete actions (TurnLeft, Stay, TurnRight)
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Observation space: oij = (i, j, ρij, αij) for each target
        # obs_dim = n_targets × 4
        self.obs_per_target = 4
        self.obs_dim = self.n_targets * self.obs_per_target
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # State dimension for centralized training
        # sensors: (x, y, angle) × n_sensors + targets: (x, y) × n_targets
        self.state_dim = self.n_sensors * 3 + self.n_targets * 2
        
        # State variables
        self.sensor_positions = None
        self.sensor_angles = None
        self.target_positions = None
        self.target_velocities = None
        self.goal_map = None  # n × m binary matrix
        self.current_step = 0
        
        # Rendering
        self.fig = None
        self.ax = None
    
    def _get_scenario1_sensor_positions(self) -> np.ndarray:
        """
        Scenario 1: Sensors at centers of cells 1, 3, 5, 7, 9.
        Grid layout:  7 8 9
                      4 5 6
                      1 2 3
        """
        cell_centers = {1: (0, 0), 3: (2, 0), 5: (1, 1), 7: (0, 2), 9: (2, 2)}
        positions = []
        for cell_num in [1, 3, 5, 7, 9]:
            col, row = cell_centers[cell_num]
            x = (col + 0.5) * self.cell_size
            y = (row + 0.5) * self.cell_size
            positions.append([x, y])
        return np.array(positions[:self.n_sensors])
    
    def _get_scenario2_sensor_positions(self) -> np.ndarray:
        """Scenario 2: Probabilistic placement with 50% threshold."""
        positions = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.np_random.random() < 0.5:
                    x = (col + 0.5) * self.cell_size
                    y = (row + 0.5) * self.cell_size
                    positions.append([x, y])
        
        positions = np.array(positions) if positions else np.zeros((0, 2))
        
        # Ensure exactly n_sensors
        if len(positions) > self.n_sensors:
            indices = self.np_random.choice(len(positions), self.n_sensors, replace=False)
            positions = positions[indices]
        elif len(positions) < self.n_sensors:
            while len(positions) < self.n_sensors:
                col = self.np_random.integers(0, self.grid_size)
                row = self.np_random.integers(0, self.grid_size)
                x = (col + 0.5) * self.cell_size
                y = (row + 0.5) * self.cell_size
                new_pos = np.array([[x, y]])
                positions = np.vstack([positions, new_pos]) if len(positions) > 0 else new_pos
        
        return positions[:self.n_sensors]
    
    def _initialize_sensor_positions(self) -> np.ndarray:
        if self.scenario == 1:
            return self._get_scenario1_sensor_positions()
        else:
            return self._get_scenario2_sensor_positions()
    
    def _initialize_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize randomly moving targets with unpredictable paths."""
        margin = self.cell_size * 0.1
        positions = self.np_random.uniform(margin, self.field_size - margin, (self.n_targets, 2))
        
        speeds = self.np_random.uniform(self.target_speed_range[0], self.target_speed_range[1], self.n_targets)
        angles = self.np_random.uniform(0, 2 * np.pi, self.n_targets)
        velocities = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)
        
        return positions, velocities
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[np.ndarray], dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.sensor_positions = self._initialize_sensor_positions()
        self.sensor_angles = self.np_random.uniform(0, 2 * np.pi, self.n_sensors)
        self.target_positions, self.target_velocities = self._initialize_targets()
        self.goal_map = np.zeros((self.n_sensors, self.n_targets), dtype=np.int32)
        self.current_step = 0
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool, bool, dict]:
        """Execute actions: TurnLeft (-5°), Stay (0°), TurnRight (+5°)."""
        assert len(actions) == self.n_sensors
        
        # Apply rotation actions
        for i, action in enumerate(actions):
            if action == self.ACTION_TURN_LEFT:
                delta = -self.rotation_step
            elif action == self.ACTION_TURN_RIGHT:
                delta = self.rotation_step
            else:
                delta = 0
            self.sensor_angles[i] = (self.sensor_angles[i] + delta) % (2 * np.pi)
        
        # Update targets (random movement)
        self._update_targets()
        self._update_goal_map()
        
        self.current_step += 1
        
        reward, info = self._calculate_reward()
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        observations = self._get_observations()
        info.update(self._get_info())
        
        return observations, reward, terminated, truncated, info
    
    def _update_targets(self):
        """Update target positions with random velocities and bouncing."""
        self.target_positions += self.target_velocities
        
        # Bounce off boundaries
        for i in range(self.n_targets):
            for d in range(2):
                if self.target_positions[i, d] < 0:
                    self.target_positions[i, d] = -self.target_positions[i, d]
                    self.target_velocities[i, d] = -self.target_velocities[i, d]
                elif self.target_positions[i, d] > self.field_size:
                    self.target_positions[i, d] = 2 * self.field_size - self.target_positions[i, d]
                    self.target_velocities[i, d] = -self.target_velocities[i, d]
        
        # Random direction changes (unpredictable paths)
        for i in range(self.n_targets):
            if self.np_random.random() < 0.1:
                angle = self.np_random.uniform(0, 2 * np.pi)
                speed = np.linalg.norm(self.target_velocities[i])
                self.target_velocities[i] = [speed * np.cos(angle), speed * np.sin(angle)]
    
    def _is_target_tracked(self, sensor_idx: int, target_idx: int) -> bool:
        """Check if target is within sensor's FoV (ρ ≤ ρmax and |α| ≤ αmax/2)."""
        sensor_pos = self.sensor_positions[sensor_idx]
        sensor_angle = self.sensor_angles[sensor_idx]
        target_pos = self.target_positions[target_idx]
        
        diff = target_pos - sensor_pos
        rho = np.linalg.norm(diff)
        
        if rho > self.sensing_range:
            return False
        
        angle_to_target = np.arctan2(diff[1], diff[0])
        alpha = self._normalize_angle(angle_to_target - sensor_angle)
        
        return abs(alpha) <= self.fov_angle / 2
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _update_goal_map(self):
        """Update goal map (n × m binary matrix): gij = 1 if target j tracked by sensor i."""
        self.goal_map = np.zeros((self.n_sensors, self.n_targets), dtype=np.int32)
        for i in range(self.n_sensors):
            for j in range(self.n_targets):
                if self._is_target_tracked(i, j):
                    self.goal_map[i, j] = 1
    
    def _calculate_reward(self) -> Tuple[float, dict]:
        """
        Calculate team reward: maximize number of tracked targets.
        
        Improved reward shaping for better learning signal:
        - Base reward: coverage rate (0-1)
        - Bonus for full coverage
        - Incremental reward for maintaining/improving coverage
        """
        targets_tracked = np.any(self.goal_map, axis=0)
        n_tracked = np.sum(targets_tracked)
        coverage_rate = n_tracked / self.n_targets
        
        # Base reward: coverage rate
        reward = coverage_rate
        
        # Bonus for full coverage (encourages complete tracking)
        if n_tracked == self.n_targets:
            reward += 0.5
        
        # Small bonus for high coverage (encourages improvement)
        if coverage_rate >= 0.8:
            reward += 0.1
        
        # Penalty for very low coverage (encourages exploration)
        if coverage_rate < 0.3:
            reward -= 0.1
        
        info = {"n_tracked": n_tracked, "coverage_rate": coverage_rate, "goal_map": self.goal_map.copy()}
        return reward, info
    
    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents: oi = (oi1, oi2, ..., oim)."""
        observations = []
        for i in range(self.n_sensors):
            obs = self._get_agent_observation(i)
            observations.append(obs)
        return observations
    
    def _get_agent_observation(self, sensor_idx: int) -> np.ndarray:
        """
        Get observation for sensor i: oij = (i, j, ρij, αij).
        - i = sensor ID (normalized)
        - j = target ID (normalized)
        - ρij = absolute distance (normalized)
        - αij = relative angle (normalized)
        """
        obs = []
        sensor_pos = self.sensor_positions[sensor_idx]
        sensor_angle = self.sensor_angles[sensor_idx]
        
        for j in range(self.n_targets):
            target_pos = self.target_positions[j]
            diff = target_pos - sensor_pos
            rho = np.linalg.norm(diff)
            angle_to_target = np.arctan2(diff[1], diff[0])
            alpha = self._normalize_angle(angle_to_target - sensor_angle)
            
            # Normalize values
            i_norm = sensor_idx / max(self.n_sensors - 1, 1)
            j_norm = j / max(self.n_targets - 1, 1)
            rho_norm = rho / (self.field_size * np.sqrt(2))
            alpha_norm = alpha / np.pi
            
            obs.extend([i_norm, j_norm, rho_norm, alpha_norm])
        
        return np.array(obs, dtype=np.float32)
    
    def get_state(self) -> np.ndarray:
        """Get global state for centralized training."""
        state = []
        for i in range(self.n_sensors):
            pos = self.sensor_positions[i] / self.field_size
            angle = self.sensor_angles[i] / (2 * np.pi)
            state.extend([pos[0], pos[1], angle])
        for j in range(self.n_targets):
            pos = self.target_positions[j] / self.field_size
            state.extend([pos[0], pos[1]])
        return np.array(state, dtype=np.float32)
    
    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "sensor_positions": self.sensor_positions.copy(),
            "sensor_angles": self.sensor_angles.copy(),
            "target_positions": self.target_positions.copy(),
            "target_velocities": self.target_velocities.copy(),
            "goal_map": self.goal_map.copy() if self.goal_map is not None else None
        }
    
    def get_avail_actions(self) -> List[np.ndarray]:
        """Get available actions (all actions always available)."""
        return [np.ones(self.n_actions, dtype=np.float32) for _ in range(self.n_sensors)]
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        
        self.ax.clear()
        self.ax.set_xlim(-2, self.field_size + 2)
        self.ax.set_ylim(-2, self.field_size + 2)
        self.ax.set_aspect('equal')
        
        targets_tracked = np.any(self.goal_map, axis=0) if self.goal_map is not None else np.zeros(self.n_targets)
        n_tracked = np.sum(targets_tracked)
        
        self.ax.set_title(
            f'Step: {self.current_step} | Tracked: {n_tracked}/{self.n_targets} ({100*n_tracked/self.n_targets:.1f}%)\n'
            f'Scenario {self.scenario}: {self.grid_size}×{self.grid_size} grid, {self.n_sensors} sensors, {self.n_targets} targets'
        )
        
        # Draw grid
        for i in range(self.grid_size + 1):
            self.ax.axhline(y=i * self.cell_size, color='lightgray', linewidth=0.5)
            self.ax.axvline(x=i * self.cell_size, color='lightgray', linewidth=0.5)
        
        self.ax.add_patch(plt.Rectangle((0, 0), self.field_size, self.field_size, fill=False, edgecolor='black', linewidth=2))
        
        # Draw sensors
        colors = plt.cm.Set1(np.linspace(0, 1, min(self.n_sensors, 10)))
        for i in range(self.n_sensors):
            pos = self.sensor_positions[i]
            angle = np.degrees(self.sensor_angles[i])
            color = colors[i % len(colors)]
            
            wedge = Wedge(pos, self.sensing_range, angle - np.degrees(self.fov_angle) / 2,
                         angle + np.degrees(self.fov_angle) / 2, alpha=0.2, color=color)
            self.ax.add_patch(wedge)
            self.ax.plot(pos[0], pos[1], 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
            
            arrow_len = self.sensing_range * 0.3
            dx = arrow_len * np.cos(self.sensor_angles[i])
            dy = arrow_len * np.sin(self.sensor_angles[i])
            self.ax.arrow(pos[0], pos[1], dx, dy, head_width=1, head_length=0.5, fc=color, ec='black')
        
        # Draw targets
        for j in range(self.n_targets):
            pos = self.target_positions[j]
            tracked = targets_tracked[j] if len(targets_tracked) > j else False
            color = 'green' if tracked else 'red'
            marker = '*' if tracked else 'x'
            self.ax.plot(pos[0], pos[1], marker, color=color, markersize=12, markeredgewidth=2)
        
        tracked_patch = mpatches.Patch(color='green', label=f'Tracked ({n_tracked})')
        untracked_patch = mpatches.Patch(color='red', label=f'Untracked ({self.n_targets - n_tracked})')
        self.ax.legend(handles=[tracked_patch, untracked_patch], loc='upper right')
        
        plt.tight_layout()
        
        if self.render_mode == "human":
            plt.pause(0.1)
            plt.draw()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            # Convert ARGB to RGB
            img = buf[:, :, [1, 2, 3]]
            return img
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def make_env(scenario: int = 1, **kwargs) -> DSNEnv:
    """Create DSN environment."""
    return DSNEnv(scenario=scenario, **kwargs)

