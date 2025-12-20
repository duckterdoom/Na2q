"""
Directional Sensor Network (DSN) Environment.

A Gymnasium-based environment for multi-agent target tracking:
- Sensors with directional FoV track randomly moving targets
- Dec-POMDP formulation for MARL training
- Supports Scenario 1 (small) and Scenario 2 (large)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches


# =============================================================================
# DSN Environment
# =============================================================================

class DSNEnv(gym.Env):
    """Directional Sensor Network Environment for Target Tracking."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Actions
    ACTION_TURN_LEFT = 0
    ACTION_STAY = 1
    ACTION_TURN_RIGHT = 2
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def __init__(
        self,
        scenario: int = 1,
        n_sensors: Optional[int] = None,
        n_targets: Optional[int] = None,
        grid_size: Optional[int] = None,
        cell_size: float = 20.0,
        sensing_range: float = 18.0,
        fov_angle: float = 90.0,  # Matches HiT-MAC: ±45° = 90° total
        rotation_step: float = 5.0,
        max_steps: int = 100,
        target_speed_range: Tuple[float, float] = (1.2, 2.4),  # Matches HiT-MAC: 2-4% of field/step
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
        self.difficulty_level = 1.0
        self.use_realistic_obs = False  # Curriculum: starts global, becomes realistic
        
        # Configure scenario
        if scenario == 1:
            self.grid_size = grid_size or 3
            self.n_sensors = n_sensors or 5
            self.n_targets = n_targets or 6
        elif scenario == 2:
            self.grid_size = grid_size or 10
            self.n_sensors = n_sensors or 50
            self.n_targets = n_targets or 60
        else:
            self.grid_size = grid_size or 3
            self.n_sensors = n_sensors or 5
            self.n_targets = n_targets or 6
        
        self.field_size = self.grid_size * self.cell_size
        self.np_random = np.random.default_rng(seed)
        
        # Spaces
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)
        
        self.obs_per_target = 4
        self.obs_dim = self.n_targets * self.obs_per_target
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.state_dim = self.n_sensors * 3 + self.n_targets * 2
        
        # State
        self.sensor_positions = None
        self.sensor_angles = None
        self.target_positions = None
        self.target_velocities = None
        self.goal_map = None
        self.current_step = 0
        
        # Rendering
        self.fig = None
        self.ax = None
    
    # -------------------------------------------------------------------------
    # Sensor Placement
    # -------------------------------------------------------------------------
    
    def _initialize_sensor_positions(self) -> np.ndarray:
        if self.scenario == 1:
            return self._get_scenario1_positions()
        else:
            return self._get_scenario2_positions()
    
    def _get_scenario1_positions(self) -> np.ndarray:
        """Scenario 1: Sensors at cells 1, 3, 5, 7, 9."""
        cell_centers = {1: (0, 0), 3: (2, 0), 5: (1, 1), 7: (0, 2), 9: (2, 2)}
        positions = []
        for cell_num in [1, 3, 5, 7, 9]:
            col, row = cell_centers[cell_num]
            x = (col + 0.5) * self.cell_size
            y = (row + 0.5) * self.cell_size
            positions.append([x, y])
        return np.array(positions[:self.n_sensors])
    
    def _get_scenario2_positions(self) -> np.ndarray:
        """Scenario 2: Random placement with 50% probability."""
        positions = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.np_random.random() < 0.5:
                    x = (col + 0.5) * self.cell_size
                    y = (row + 0.5) * self.cell_size
                    positions.append([x, y])
        
        positions = np.array(positions) if positions else np.zeros((0, 2))
        
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
    
    # -------------------------------------------------------------------------
    # Target Management
    # -------------------------------------------------------------------------
    
    def set_curriculum_difficulty(self, level: float):
        """Set curriculum difficulty.
        
        0-49%: Old speed + global observations (easy)
        50-100%: HiT-MAC speed + realistic observations (hard)
        """
        self.difficulty_level = np.clip(level, 0.0, 1.0)
        # Switch to realistic observations at 50% curriculum
        self.use_realistic_obs = level >= 0.5
    
    def _initialize_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize randomly moving targets."""
        margin = self.cell_size * 0.1
        positions = self.np_random.uniform(margin, self.field_size - margin, (self.n_targets, 2))
        
        # Scale speed: old speed (0.3-0.7) → HiT-MAC (1.2-2.4) at 50% curriculum
        speed_progress = min(1.0, self.difficulty_level * 2)  # 0.5 → 1.0, caps at 1.0
        # Old speed: (0.3, 0.7), HiT-MAC: (1.2, 2.4)
        min_speed = 0.3 + (self.target_speed_range[0] - 0.3) * speed_progress
        max_speed = 0.7 + (self.target_speed_range[1] - 0.7) * speed_progress
        
        speeds = self.np_random.uniform(min_speed, max_speed, self.n_targets)
        angles = self.np_random.uniform(0, 2 * np.pi, self.n_targets)
        velocities = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)
        
        return positions, velocities
    
    def _update_targets(self):
        """Update target positions with bouncing and speed variation."""
        # Apply 20% speed variation (matches DSN)
        speed_variation = 1 + 0.2 * self.np_random.random(self.n_targets)
        varied_velocities = self.target_velocities * speed_variation[:, np.newaxis]
        
        self.target_positions += varied_velocities
        
        for i in range(self.n_targets):
            for d in range(2):
                if self.target_positions[i, d] < 0:
                    self.target_positions[i, d] = -self.target_positions[i, d]
                    self.target_velocities[i, d] = -self.target_velocities[i, d]
                elif self.target_positions[i, d] > self.field_size:
                    self.target_positions[i, d] = 2 * self.field_size - self.target_positions[i, d]
                    self.target_velocities[i, d] = -self.target_velocities[i, d]
        
        # Random direction changes (10% chance per target)
        for i in range(self.n_targets):
            if self.np_random.random() < 0.1:
                angle = self.np_random.uniform(0, 2 * np.pi)
                speed = np.linalg.norm(self.target_velocities[i])
                self.target_velocities[i] = [speed * np.cos(angle), speed * np.sin(angle)]
    
    # -------------------------------------------------------------------------
    # Core Environment Methods
    # -------------------------------------------------------------------------
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[np.ndarray], dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.sensor_positions = self._initialize_sensor_positions()
        self.target_positions, self.target_velocities = self._initialize_targets()
        
        # Initialize angles to point at nearest target
        angles = []
        for i in range(self.n_sensors):
            dists = [np.linalg.norm(self.target_positions[j] - self.sensor_positions[i]) 
                     for j in range(self.n_targets)]
            nearest_idx = np.argmin(dists)
            diff = self.target_positions[nearest_idx] - self.sensor_positions[i]
            angles.append(np.arctan2(diff[1], diff[0]))
        self.sensor_angles = np.array(angles)
        
        self.goal_map = np.zeros((self.n_sensors, self.n_targets), dtype=np.int32)
        self.current_step = 0
        
        return self._get_observations(), self._get_info()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool, bool, dict]:
        """Execute actions and return next state."""
        if len(actions) != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} actions, got {len(actions)}")
        
        # Apply rotations
        for i, action in enumerate(actions):
            if action == self.ACTION_TURN_LEFT:
                self.sensor_angles[i] -= self.rotation_step
            elif action == self.ACTION_TURN_RIGHT:
                self.sensor_angles[i] += self.rotation_step
            self.sensor_angles[i] = self.sensor_angles[i] % (2 * np.pi)
        
        self._update_targets()
        self._update_goal_map()
        self.current_step += 1
        
        reward, info = self._calculate_reward()
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_observations(), reward, terminated, truncated, info
    
    # -------------------------------------------------------------------------
    # Tracking Logic
    # -------------------------------------------------------------------------
    
    def _is_target_tracked(self, sensor_idx: int, target_idx: int) -> bool:
        """Check if target is within sensor's FoV."""
        diff = self.target_positions[target_idx] - self.sensor_positions[sensor_idx]
        rho = np.linalg.norm(diff)
        
        if rho > self.sensing_range:
            return False
        
        angle_to_target = np.arctan2(diff[1], diff[0])
        alpha = self._normalize_angle(angle_to_target - self.sensor_angles[sensor_idx])
        
        return abs(alpha) <= self.fov_angle / 2
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _update_goal_map(self):
        """Update goal map: gij = 1 if target j tracked by sensor i."""
        self.goal_map = np.zeros((self.n_sensors, self.n_targets), dtype=np.int32)
        for i in range(self.n_sensors):
            for j in range(self.n_targets):
                if self._is_target_tracked(i, j):
                    self.goal_map[i, j] = 1
    
    # -------------------------------------------------------------------------
    # Reward Calculation
    # -------------------------------------------------------------------------
    
    def _calculate_reward(self) -> Tuple[float, dict]:
        """Calculate team reward based on coverage."""
        targets_tracked = np.any(self.goal_map, axis=0)
        n_tracked = np.sum(targets_tracked)
        coverage_rate = n_tracked / self.n_targets
        
        # Base reward = coverage rate
        reward = coverage_rate
        
        # Bonuses for high coverage
        if n_tracked == self.n_targets:
            reward += 1.0
        elif coverage_rate >= 0.8:
            reward += 0.6
        elif coverage_rate >= 0.5:
            reward += 0.2
        
        # Centering bonus (reward for pointing at targets)
        centering_bonus = 0.0
        for i in range(self.n_sensors):
            dists = [np.linalg.norm(self.target_positions[j] - self.sensor_positions[i]) 
                     for j in range(self.n_targets)]
            nearest_idx = np.argmin(dists)
            
            diff = self.target_positions[nearest_idx] - self.sensor_positions[i]
            angle_to_target = np.arctan2(diff[1], diff[0])
            angle_diff = (angle_to_target - self.sensor_angles[i] + np.pi) % (2 * np.pi) - np.pi
            
            alignment_quality = np.cos(angle_diff)
            centering_bonus += alignment_quality * (1.0 / self.n_sensors)
        
        reward += centering_bonus
        
        info = {"n_tracked": n_tracked, "coverage_rate": coverage_rate, "goal_map": self.goal_map.copy()}
        return reward, info
    
    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------
    
    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents."""
        return [self._get_agent_observation(i) for i in range(self.n_sensors)]
    
    def _get_agent_observation(self, sensor_idx: int) -> np.ndarray:
        """Get observation for sensor i.
        
        If use_realistic_obs=True: only observe targets within sensing range AND FoV.
        If use_realistic_obs=False: observe all targets (global view).
        """
        sensor_pos = self.sensor_positions[sensor_idx]
        sensor_angle = self.sensor_angles[sensor_idx]
        
        target_obs_list = []
        for j in range(self.n_targets):
            diff = self.target_positions[j] - sensor_pos
            rho = np.linalg.norm(diff)
            alpha = self._normalize_angle(np.arctan2(diff[1], diff[0]) - sensor_angle)
            
            i_norm = sensor_idx / max(self.n_sensors - 1, 1)
            j_norm = j / max(self.n_targets - 1, 1)
            
            if self.use_realistic_obs:
                # Realistic: only observe within sensing range AND FoV
                in_range = rho <= self.sensing_range
                in_fov = abs(alpha) <= self.fov_angle / 2
                is_visible = in_range and in_fov
                
                if is_visible:
                    rho_norm = rho / (self.field_size * np.sqrt(2))
                    alpha_norm = alpha / np.pi
                else:
                    rho_norm = 0.0
                    alpha_norm = 0.0
                sort_key = rho if is_visible else float('inf')
            else:
                # Global: observe all targets
                rho_norm = rho / (self.field_size * np.sqrt(2))
                alpha_norm = alpha / np.pi
                sort_key = rho
            
            target_obs_list.append((sort_key, [i_norm, j_norm, rho_norm, alpha_norm]))
        
        # Sort by distance (visible first)
        target_obs_list.sort(key=lambda x: x[0])
        
        obs = []
        for _, obs_values in target_obs_list:
            obs.extend(obs_values)
        
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
        """Get available actions (all always available)."""
        return [np.ones(self.n_actions, dtype=np.float32) for _ in range(self.n_sensors)]
    
    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    
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
            f'Scenario {self.scenario}: {self.grid_size}×{self.grid_size} grid'
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
            self.ax.plot(pos[0], pos[1], 'o', color=color, markersize=8, markeredgecolor='black')
        
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
            return buf[:, :, [1, 2, 3]]
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# =============================================================================
# Factory Function
# =============================================================================

def make_env(scenario: int = 1, **kwargs) -> DSNEnv:
    """Create DSN environment."""
    return DSNEnv(scenario=scenario, **kwargs)
