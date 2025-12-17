"""
NA²Q Agent - Training and Inference Wrapper.

Wraps the NA2Q model with training loop, action selection,
and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
import numpy as np
from typing import Tuple, Optional, Dict

from .na2q_model import NA2Q


# =============================================================================
# NA²Q Agent
# =============================================================================

class NA2QAgent:
    """
    NA²Q Agent for training and inference.
    
    Uses RMSprop for Q-learning and Adam for VAE.
    Implements epsilon-greedy exploration and Double DQN.
    """
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        semantic_hidden_dim: int = 32,
        latent_dim: int = 16,
        attention_hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        target_update_interval: int = 200,
        vae_loss_weight: float = 0.1,
        device: str = "cpu",
        use_amp: bool = True
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.vae_loss_weight = vae_loss_weight
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        
        # Networks
        self.model = NA2Q(
            n_agents, obs_dim, state_dim, n_actions, hidden_dim, rnn_hidden_dim,
            semantic_hidden_dim, latent_dim, attention_hidden_dim
        ).to(self.device)
        
        self.target_model = NA2Q(
            n_agents, obs_dim, state_dim, n_actions, hidden_dim, rnn_hidden_dim,
            semantic_hidden_dim, latent_dim, attention_hidden_dim
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        # Optimizers
        q_params = list(self.model.agent_q_network.parameters()) + list(self.model.mixer.parameters())
        vae_params = list(self.model.semantics_encoder.parameters())
        
        self.q_optimizer = torch.optim.RMSprop(q_params, lr=lr, alpha=0.99, eps=1e-5)
        self.vae_optimizer = torch.optim.Adam(vae_params, lr=lr, betas=(0.9, 0.999))
        
        self.q_scheduler = torch.optim.lr_scheduler.StepLR(self.q_optimizer, step_size=15000, gamma=0.95)
        self.vae_scheduler = torch.optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=15000, gamma=0.95)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.train_step = 0
        self.hidden_states = None
    
    def init_hidden(self, batch_size: int = 1):
        self.hidden_states = self.model.init_hidden(batch_size).to(self.device)
    
    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------
    
    def select_actions(
        self,
        observations: np.ndarray,
        prev_actions: Optional[np.ndarray] = None,
        avail_actions: Optional[np.ndarray] = None,
        evaluate: bool = False
    ) -> np.ndarray:
        """Select actions using ε-greedy policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            
            prev_actions_tensor = None
            if prev_actions is not None:
                prev_actions_tensor = torch.LongTensor(prev_actions).unsqueeze(0).to(self.device)
            
            if self.hidden_states is None:
                self.init_hidden(1)
            
            q_values, self.hidden_states = self.model.get_q_values(
                obs_tensor, self.hidden_states, prev_actions_tensor
            )
            q_values = q_values.squeeze(0)
            
            if avail_actions is not None:
                avail_mask = torch.FloatTensor(avail_actions).to(self.device)
                q_values[avail_mask == 0] = -float('inf')
            
            if not evaluate and np.random.random() < self.epsilon:
                if avail_actions is not None:
                    actions = []
                    for i in range(self.n_agents):
                        avail = np.where(avail_actions[i] == 1)[0]
                        actions.append(np.random.choice(avail))
                    actions = np.array(actions)
                else:
                    actions = np.random.randint(0, self.n_actions, self.n_agents)
            else:
                actions = q_values.argmax(dim=-1).cpu().numpy()
        
        return actions
    
    def select_actions_batch(
        self,
        observations: np.ndarray,
        prev_actions: Optional[np.ndarray] = None,
        avail_actions: Optional[np.ndarray] = None,
        hidden_states: Optional[torch.Tensor] = None,
        evaluate: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Select actions for batch of environments."""
        batch_size = observations.shape[0]
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            
            prev_actions_tensor = None
            if prev_actions is not None:
                prev_actions_tensor = torch.LongTensor(prev_actions).to(self.device)
            
            if hidden_states is None:
                hidden_states = self.model.init_hidden(batch_size).to(self.device)
            
            q_values, hidden_states = self.model.get_q_values(
                obs_tensor, hidden_states, prev_actions_tensor
            )
            
            if avail_actions is not None:
                avail_mask = torch.FloatTensor(avail_actions).to(self.device)
                q_values[avail_mask == 0] = -float('inf')
            
            if not evaluate and np.random.random() < self.epsilon:
                if avail_actions is not None:
                    actions = np.zeros((batch_size, self.n_agents), dtype=np.int64)
                    for b in range(batch_size):
                        for i in range(self.n_agents):
                            avail = np.where(avail_actions[b, i] == 1)[0]
                            actions[b, i] = np.random.choice(avail)
                else:
                    actions = np.random.randint(0, self.n_actions, (batch_size, self.n_agents))
            else:
                actions = q_values.argmax(dim=-1).cpu().numpy()
        
        return actions, hidden_states
    
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    
    def update_epsilon(self):
        """Update epsilon with linear decay."""
        if self.train_step < self.epsilon_decay:
            self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * (1.0 - self.train_step / self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_end
    
    def soft_update_target(self, tau: float = 0.005):
        """Soft target update using Polyak averaging."""
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    
    def train_step_fn(self, batch: dict) -> dict:
        """Perform one training step."""
        device = self.device
        
        # Unpack batch
        observations = torch.as_tensor(batch["observations"], device=device, dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], device=device, dtype=torch.long)
        rewards = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32)
        states = torch.as_tensor(batch["states"], device=device, dtype=torch.float32)
        next_observations = torch.as_tensor(batch["next_observations"], device=device, dtype=torch.float32)
        next_states = torch.as_tensor(batch["next_states"], device=device, dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], device=device, dtype=torch.float32)
        avail_actions = torch.as_tensor(batch["avail_actions"], device=device, dtype=torch.float32)
        
        batch_size = observations.size(0)
        seq_len = observations.size(1)
        n_agents = self.n_agents
        
        hidden = self.model.init_hidden(batch_size).to(device)
        target_hidden = self.target_model.init_hidden(batch_size).to(device)
        
        q_totals, target_q_totals, vae_losses = [], [], []
        
        # Sequence processing
        device_type = "cuda" if device.type == "cuda" else "cpu"
        
        with torch.amp.autocast(device_type, enabled=self.use_amp):
            for t in range(seq_len):
                obs_t = observations[:, t]
                state_t = states[:, t]
                actions_t = actions[:, t]
                
                # Done handling
                done_t = dones[:, t] if dones.dim() == 2 else torch.zeros(batch_size, device=device)
                prev_done = dones[:, t-1] if t > 0 and dones.dim() == 2 else torch.zeros(batch_size, device=device)
                
                # Reset hidden for done episodes
                prev_done_expanded = prev_done.repeat_interleave(n_agents).view(batch_size * n_agents, 1)
                hidden = hidden * (1 - prev_done_expanded.expand_as(hidden).float())
                target_hidden = target_hidden * (1 - prev_done_expanded.expand_as(target_hidden).float())
                
                # Previous actions
                prev_actions_t = actions[:, t-1] if t > 0 else torch.zeros_like(actions_t)
                
                # Forward pass
                result = self.model(obs_t, hidden, state_t, actions_t, prev_actions_t)
                q_totals.append(result['q_total'])
                vae_losses.append(result['vae_loss'])
                hidden = result['hidden_states']
                
                # Target computation (Double DQN)
                with torch.no_grad():
                    next_obs_t = next_observations[:, min(t, seq_len-1)]
                    next_state_t = next_states[:, min(t, seq_len-1)]
                    
                    done_t_expanded = done_t.repeat_interleave(n_agents).view(batch_size * n_agents, 1)
                    target_hidden = target_hidden * (1 - done_t_expanded.expand_as(target_hidden).float())
                    
                    online_q_values, _ = self.model.get_q_values(next_obs_t, hidden.detach(), actions_t)
                    next_actions = online_q_values.argmax(dim=-1)
                    
                    target_result = self.target_model(next_obs_t, target_hidden, next_state_t, next_actions, actions_t)
                    target_q_totals.append(target_result['q_total'])
                    target_hidden = target_result['hidden_states']
            
            q_totals = torch.stack(q_totals, dim=1).squeeze(-1)
            target_q_totals = torch.stack(target_q_totals, dim=1).squeeze(-1)
            vae_loss = torch.stack(vae_losses).mean()
        
        # Reshape rewards/dones
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)
        if dones.dim() == 1:
            dones = dones.unsqueeze(0)
        
        if rewards.size(0) != target_q_totals.size(0):
            rewards = rewards.expand(target_q_totals.size(0), -1)
            dones = dones.expand(target_q_totals.size(0), -1)
        
        # TD targets
        targets = rewards + self.gamma * (1 - dones) * target_q_totals
        
        # Losses
        q_totals_clipped = torch.clamp(q_totals, -100.0, 400.0)
        targets_clipped = torch.clamp(targets.detach(), -100.0, 400.0)
        td_loss = F.smooth_l1_loss(q_totals_clipped, targets_clipped)
        
        warmup_steps = 5000
        vae_weight = self.vae_loss_weight * min(1.0, self.train_step / warmup_steps)
        total_loss = td_loss + vae_weight * vae_loss
        
        # Optimization
        self.q_optimizer.zero_grad(set_to_none=True)
        self.vae_optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.q_optimizer)
            self.scaler.unscale_(self.vae_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.q_optimizer)
            self.scaler.step(self.vae_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.q_optimizer.step()
            self.vae_optimizer.step()
        
        self.train_step += 1
        self.soft_update_target(tau=0.008)
        self.q_scheduler.step()
        self.vae_scheduler.step()
        self.update_epsilon()
        
        return {
            "loss": total_loss.item(),
            "td_loss": td_loss.item(),
            "vae_loss": vae_loss.item(),
            "epsilon": self.epsilon
        }
    
    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
            "train_step": self.train_step,
            "epsilon": self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.vae_optimizer.load_state_dict(checkpoint["vae_optimizer_state_dict"])
        self.train_step = checkpoint["train_step"]
        self.epsilon = checkpoint["epsilon"]
    
    # -------------------------------------------------------------------------
    # Interpretability
    # -------------------------------------------------------------------------
    
    def get_interpretable_contributions(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get interpretable contributions for visualization."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(0).to(self.device)
            
            if self.hidden_states is None:
                self.init_hidden(1)
            
            result = self.model(obs_tensor, self.hidden_states, state_tensor, actions_tensor)
            
            return {
                'individual_contribs': result['individual_contribs'].squeeze(0).cpu().numpy(),
                'pairwise_contribs': result['pairwise_contribs'].squeeze(0).cpu().numpy(),
                'attention_weights': result['attention_weights'].squeeze(0).cpu().numpy(),
                'masks': result['masks'].squeeze(0).cpu().numpy(),
                'q_total': result['q_total'].squeeze().cpu().numpy()
            }
