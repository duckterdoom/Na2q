"""
NA2Q Agent Implementation.
Main agent class combining all components (Q-Network, VAE, Mixer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
import numpy as np
from typing import Tuple, Optional, Dict, List

from .components.network import AgentQNetwork
from .components.encoder import IdentitySemanticsEncoder
from .components.mixer import NA2QMixer


class NA2Q(nn.Module):
    """
    Complete NA²Q Model.
    Combines Agent Q-Networks (GRU), Identity Semantics (VAE), and NA²Q Mixer (GAM).
    """
    
    def __init__(self, n_agents: int, obs_dim: int, state_dim: int, n_actions: int,
                 hidden_dim: int = 64, rnn_hidden_dim: int = 64, semantic_hidden_dim: int = 32,
                 latent_dim: int = 16, attention_hidden_dim: int = 64):
        super().__init__()
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        
        self.agent_q_network = AgentQNetwork(obs_dim, n_actions, hidden_dim, rnn_hidden_dim)
        # Identity semantics uses hidden states (rnn_hidden_dim), not observations (Figure 2)
        self.semantics_encoder = IdentitySemanticsEncoder(rnn_hidden_dim, semantic_hidden_dim, latent_dim)
        self.mixer = NA2QMixer(n_agents, state_dim, latent_dim, attention_hidden_dim)
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.agent_q_network.init_hidden(batch_size * self.n_agents)
    
    def forward(self, observations: torch.Tensor, hidden_states: torch.Tensor,
                state: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None,
                prev_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            observations: [batch, n_agents, obs_dim]
            hidden_states: [batch * n_agents, rnn_hidden_dim]
            state: [batch, state_dim]
            actions: [batch, n_agents]
            prev_actions: [batch, n_agents]
        """
        batch_size = observations.size(0)
        
        # Process through Q-network (Figure 2: o_i^t, u_i^{t-1} → MLP → GRU → MLP → Q_i, h_i^t)
        obs_flat = observations.reshape(batch_size * self.n_agents, self.obs_dim)
        prev_actions_flat = None
        if prev_actions is not None:
            prev_actions_flat = prev_actions.reshape(batch_size * self.n_agents)
        q_values_flat, hidden_states = self.agent_q_network(obs_flat, hidden_states, prev_actions_flat)
        q_values = q_values_flat.reshape(batch_size, self.n_agents, self.n_actions)
        
        # Process through semantics encoder (Figure 2: h_i^t → E_ω1 → z_i → D_ω2 → reconstruction)
        # Extract hidden states for each agent: [batch, n_agents, rnn_hidden_dim]
        hidden_states_reshaped = hidden_states.view(batch_size, self.n_agents, -1)
        hidden_flat = hidden_states_reshaped.reshape(batch_size * self.n_agents, -1)
        z_flat, mask_flat, recon_flat, mean_flat, logvar_flat = self.semantics_encoder(hidden_flat)
        agent_semantics = z_flat.view(batch_size, self.n_agents, -1)
        masks = mask_flat.view(batch_size, self.n_agents, -1)
        recons = recon_flat.view(batch_size, self.n_agents, -1)
        means = mean_flat.view(batch_size, self.n_agents, -1)
        logvars = logvar_flat.view(batch_size, self.n_agents, -1)
        
        # VAE loss: MSE(recon, h_i^t) + β×KL (β=0.1) - reconstructing hidden states (Figure 2)
        recon_loss = F.mse_loss(recon_flat, hidden_flat, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar_flat - mean_flat.pow(2) - logvar_flat.exp())
        vae_loss = recon_loss + 0.1 * kl_loss
        
        result = {
            'q_values': q_values, 'hidden_states': hidden_states, 'masks': masks,
            'agent_semantics': agent_semantics, 'vae_loss': vae_loss
        }
        
        # Compute Q_total if state and actions provided
        # Implements GAM-based value decomposition from Section 4:
        # Q_tot(s, a) = Σ_k (α_k(s, z) × f_k(Q(s, a))) + bias(s)
        # where Q(s, a) = [Q_1(s_1, a_1), ..., Q_n(s_n, a_n)] are chosen Q-values
        if state is not None and actions is not None:
            # Extract Q-values for chosen actions: Q_i(s_i, a_i) for each agent i
            actions_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()
            chosen_q_values = (q_values * actions_onehot).sum(dim=-1)  # [batch, n_agents]
            
            # Apply mixer: Q_tot = Σ_k (α_k × f_k(Q)) + bias(s)
            # This implements the GAM framework from Section 4
            q_total, attention_weights, shape_outputs = self.mixer(chosen_q_values, state, agent_semantics)
            individual_contribs, pairwise_contribs = self.mixer.get_individual_contributions(attention_weights, shape_outputs)
            
            result.update({
                'q_total': q_total, 'attention_weights': attention_weights,
                'shape_outputs': shape_outputs, 'individual_contribs': individual_contribs,
                'pairwise_contribs': pairwise_contribs
            })
        
        return result
    
    def get_q_values(self, observations: torch.Tensor, hidden_states: torch.Tensor,
                     prev_actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = observations.size(0)
        obs_flat = observations.reshape(batch_size * self.n_agents, self.obs_dim)
        
        prev_actions_flat = None
        if prev_actions is not None:
            prev_actions_flat = prev_actions.reshape(batch_size * self.n_agents)
            
        q_values_flat, hidden_states = self.agent_q_network(obs_flat, hidden_states, prev_actions_flat)
        q_values = q_values_flat.reshape(batch_size, self.n_agents, self.n_actions)
        return q_values, hidden_states


class NA2QAgent:
    """
    NA²Q Agent for training and inference.
    Uses RMSprop for Q-learning and Adam for VAE.
    """
    
    def __init__(self, n_agents: int, obs_dim: int, state_dim: int, n_actions: int,
                 hidden_dim: int = 64, rnn_hidden_dim: int = 64, semantic_hidden_dim: int = 32,
                 latent_dim: int = 16, attention_hidden_dim: int = 64, lr: float = 5e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 50000, target_update_interval: int = 200,  # From paper: 50k steps, 200 step updates
                 vae_loss_weight: float = 0.1, device: str = "cpu", use_amp: bool = True):
        
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
        
        # Online and target networks
        self.model = NA2Q(n_agents, obs_dim, state_dim, n_actions, hidden_dim, rnn_hidden_dim,
                         semantic_hidden_dim, latent_dim, attention_hidden_dim).to(self.device)
        self.target_model = NA2Q(n_agents, obs_dim, state_dim, n_actions, hidden_dim, rnn_hidden_dim,
                                semantic_hidden_dim, latent_dim, attention_hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Freeze target network - it should not have gradients
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        # Optimizers (RMSprop for Q, Adam for VAE)
        q_params = list(self.model.agent_q_network.parameters()) + list(self.model.mixer.parameters())
        vae_params = list(self.model.semantics_encoder.parameters())
        
        # Learning rate from paper: 0.0005 (Table 3)
        # Using paper value for exact match with NA2Q framework
        self.q_optimizer = torch.optim.RMSprop(q_params, lr=lr, alpha=0.99, eps=1e-5)
        self.vae_optimizer = torch.optim.Adam(vae_params, lr=lr, betas=(0.9, 0.999))
        
        # Learning rate schedulers optimized for long training (30k episodes)
        # Very gentle decay: reduce by only 5% every 15000 steps to maintain learning capacity
        # For 30k episodes (~100k+ steps), this ensures LR stays high enough for continued learning
        self.q_scheduler = torch.optim.lr_scheduler.StepLR(self.q_optimizer, step_size=15000, gamma=0.95)
        self.vae_scheduler = torch.optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=15000, gamma=0.95)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.train_step = 0
        self.hidden_states = None
    
    def init_hidden(self, batch_size: int = 1):
        self.hidden_states = self.model.init_hidden(batch_size).to(self.device)
    
    def select_actions(self, observations: np.ndarray, prev_actions: Optional[np.ndarray] = None,
                       avail_actions: Optional[np.ndarray] = None, evaluate: bool = False) -> np.ndarray:
        """Select actions using ε-greedy policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            
            prev_actions_tensor = None
            if prev_actions is not None:
                prev_actions_tensor = torch.LongTensor(prev_actions).unsqueeze(0).to(self.device)
            
            if self.hidden_states is None:
                self.init_hidden(1)
            
            q_values, self.hidden_states = self.model.get_q_values(obs_tensor, self.hidden_states, prev_actions_tensor)
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
    
    def select_actions_batch(self, observations: np.ndarray, prev_actions: Optional[np.ndarray] = None,
                             avail_actions: Optional[np.ndarray] = None,
                             hidden_states: Optional[torch.Tensor] = None,
                             evaluate: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Select actions for batch of environments.
        """
        batch_size = observations.shape[0]
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            
            prev_actions_tensor = None
            if prev_actions is not None:
                prev_actions_tensor = torch.LongTensor(prev_actions).to(self.device)
            
            if hidden_states is None:
                hidden_states = self.model.init_hidden(batch_size).to(self.device)
            
            q_values, hidden_states = self.model.get_q_values(obs_tensor, hidden_states, prev_actions_tensor)
            # q_values: [batch, n_agents, n_actions]
            
            if avail_actions is not None:
                avail_mask = torch.FloatTensor(avail_actions).to(self.device)
                q_values[avail_mask == 0] = -float('inf')
            
            # ε-greedy action selection
            if not evaluate and np.random.random() < self.epsilon:
                # Random actions for exploration
                if avail_actions is not None:
                    actions = np.zeros((batch_size, self.n_agents), dtype=np.int64)
                    for b in range(batch_size):
                        for i in range(self.n_agents):
                            avail = np.where(avail_actions[b, i] == 1)[0]
                            actions[b, i] = np.random.choice(avail)
                else:
                    actions = np.random.randint(0, self.n_actions, (batch_size, self.n_agents))
            else:
                # Greedy actions
                actions = q_values.argmax(dim=-1).cpu().numpy()
        
        return actions, hidden_states
    
    def update_epsilon(self):
        """Update epsilon with linear decay based on training steps."""
        if self.train_step < self.epsilon_decay:
            self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * (1.0 - self.train_step / self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_end
    
    
    def soft_update_target(self, tau: float = 0.005):
        """Soft target update using Polyak averaging."""
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    
    def train_step_fn(self, batch: dict) -> dict:
        """Perform one training step. Loss = TD_loss + β × VAE_loss."""
        device = self.device
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
        
        # Hidden states are per-agent: (batch_size * n_agents, hidden_dim)
        hidden = self.model.init_hidden(batch_size).to(device)
        target_hidden = self.target_model.init_hidden(batch_size).to(device)
        
        q_totals, target_q_totals, vae_losses = [], [], []
        dones_is_seq = dones.dim() == 2
        dones_is_vec = dones.dim() == 1
        
        def get_done(step_idx: int) -> torch.Tensor:
            """Return done flags for a given timestep as [batch]."""
            if dones_is_seq:
                return dones[:, step_idx]
            if dones_is_vec:
                val = dones[min(step_idx, dones.numel() - 1)]
                return val.expand(batch_size) if batch_size > 1 else val
            return torch.zeros(batch_size, device=device)
        
        def get_avail(step_idx: int) -> Optional[torch.Tensor]:
            """Return available-action mask for a given timestep as [batch, n_agents, n_actions] or None."""
            if avail_actions.dim() == 4:
                return avail_actions[:, step_idx]
            if avail_actions.dim() == 3:
                if avail_actions.size(0) == batch_size:
                    return avail_actions[:, step_idx]
                # Fallback for shape [seq, n_agents, n_actions]
                return avail_actions[step_idx].unsqueeze(0)
            if avail_actions.dim() == 2:
                return avail_actions.unsqueeze(0)
            return None
        
        # Use torch.amp.autocast instead of torch.cuda.amp.autocast for PyTorch 2.4+ compatibility
        # If device is CPU, autocast handles it or does nothing gracefully
        device_type = "cuda" if device.type == "cuda" else "cpu"
        
        with torch.amp.autocast(device_type, enabled=self.use_amp):
            for t in range(seq_len):
                obs_t, state_t, actions_t = observations[:, t], states[:, t], actions[:, t]
                done_t = get_done(t)
                
                # Reset hidden states for done episodes (at start of timestep)
                # Hidden states are (batch_size * n_agents, hidden_dim)
                prev_done = get_done(t - 1) if t > 0 else torch.zeros(batch_size, device=device)
                prev_done_expanded = prev_done.repeat_interleave(n_agents).view(batch_size * n_agents, 1)
                done_mask = prev_done_expanded.expand(batch_size * n_agents, hidden.size(1))
                hidden = hidden * (1 - done_mask.float())
                target_hidden = target_hidden * (1 - done_mask.float())
                
                # Use previous actions for current step (shift actions right by 1, pad with 0 at start)
                # In training batch: actions[:, t] is action at time t.
                # We need prev_action for time t, which is action at t-1.
                if t == 0:
                    prev_actions_t = torch.zeros_like(actions_t)
                else:
                    prev_actions_t = actions[:, t-1]
                
                result = self.model(obs_t, hidden, state_t, actions_t, prev_actions_t)
                q_totals.append(result['q_total'])
                vae_losses.append(result['vae_loss'])
                hidden = result['hidden_states']
                
                with torch.no_grad():
                    # Use next timestep's observations and states
                    if t < seq_len - 1:
                        next_obs_t = next_observations[:, t]
                        next_state_t = next_states[:, t]
                    else:
                        # Last timestep: use the last next_obs/next_state
                        next_obs_t = next_observations[:, -1]
                        next_state_t = next_states[:, -1]
                    next_avail = get_avail(t if t < seq_len - 1 else seq_len - 1)
                    
                    # Reset target hidden for done episodes
                    done_t_expanded = done_t.repeat_interleave(n_agents).view(batch_size * n_agents, 1)
                    done_mask_target = done_t_expanded.expand(batch_size * n_agents, target_hidden.size(1))
                    target_hidden = target_hidden * (1 - done_mask_target.float())
                    
                    # Double DQN with prev_actions
                    # For next_obs_t (time t+1), prev_action is action at time t (actions_t)
                    online_q_values, _ = self.model.get_q_values(next_obs_t, hidden.detach(), actions_t)
                    if next_avail is not None and next_avail.dim() == 3:
                        online_q_values[next_avail == 0] = -float('inf')
                    next_actions = online_q_values.argmax(dim=-1)  # Select using online network
                    
                    # Evaluate using target network
                    # Target network also needs prev_actions (actions_t)
                    target_result = self.target_model(next_obs_t, target_hidden, next_state_t, next_actions, actions_t)
                    target_q_totals.append(target_result['q_total'])
                    target_hidden = target_result['hidden_states']
            
            q_totals = torch.stack(q_totals, dim=1).squeeze(-1)  # [batch, seq]
            target_q_totals = torch.stack(target_q_totals, dim=1).squeeze(-1)  # [batch, seq]
            vae_loss = torch.stack(vae_losses).mean()
        
        # Handle rewards and dones - they come as [batch, seq] from replay buffer
        # Ensure they're 2D: [batch, seq]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)  # [1, seq]
        if dones.dim() == 1:
            dones = dones.unsqueeze(0)  # [1, seq]
        
        # Ensure batch dimensions match
        if rewards.size(0) != target_q_totals.size(0):
            if rewards.size(0) == 1:
                rewards = rewards.expand(target_q_totals.size(0), -1)
            if dones.size(0) == 1:
                dones = dones.expand(target_q_totals.size(0), -1)
        
        # Compute targets: r + γ * (1 - done) * Q_target
        targets = rewards + self.gamma * (1 - dones) * target_q_totals
        
        # Clip Q-totals to prevent divergence
        q_totals_clipped = torch.clamp(q_totals, -100.0, 400.0)
        targets_clipped = torch.clamp(targets.detach(), -100.0, 400.0)
        
        # Use Huber loss (smooth L1) for robustness
        td_loss = F.smooth_l1_loss(q_totals_clipped, targets_clipped)
        
        # VAE warmup to let Q-learning stabilize first
        warmup_steps = 5000
        vae_weight = self.vae_loss_weight * min(1.0, self.train_step / warmup_steps)
        total_loss = td_loss + vae_weight * vae_loss
        
        self.q_optimizer.zero_grad(set_to_none=True)
        self.vae_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.q_optimizer)
            self.scaler.unscale_(self.vae_optimizer)
            # Balanced gradient clipping for better learning (increased from 0.2 to 0.5)
            # Allows more learning capacity while still preventing gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.q_optimizer)
            self.scaler.step(self.vae_optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            # Balanced gradient clipping for better learning (increased from 0.2 to 0.5)
            # Allows more learning capacity while still preventing gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.q_optimizer.step()
            self.vae_optimizer.step()
        
        self.train_step += 1
        
        # Use soft target update (Polyak averaging) instead of hard copy
        # This provides more stable learning compared to periodic hard updates
        # Soft update: θ_target = τ * θ_online + (1 - τ) * θ_target
        # Reduced tau to 0.008 for better stability during long 30k episode training
        self.soft_update_target(tau=0.008)  # Slower target update for long-term stability
        
        # Update learning rate schedulers (every training step)
        # Optimized for long episode training: LR decays less aggressively (5% every 15k steps)
        self.q_scheduler.step()
        self.vae_scheduler.step()
        
        self.update_epsilon()
        
        return {
            "loss": total_loss.item(), "td_loss": td_loss.item(),
            "vae_loss": vae_loss.item(), "q_total_mean": q_totals.mean().item(),
            "epsilon": self.epsilon,
            "q_lr": self.q_optimizer.param_groups[0]['lr'],
            "vae_lr": self.vae_optimizer.param_groups[0]['lr']
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
            "train_step": self.train_step, "epsilon": self.epsilon
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
    
    def get_interpretable_contributions(self, observations: np.ndarray, state: np.ndarray,
                                        actions: np.ndarray) -> Dict[str, np.ndarray]:
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
