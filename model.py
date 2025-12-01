"""
NA²Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning

Based on ICML 2023 paper by Liu, Zhu, Chen (Nanjing University)
Paper: https://proceedings.mlr.press/v202/liu23be/liu23be.pdf
Code: https://github.com/zichuan-liu/NA2Q

Architecture (from paper):
1. GAM-based Value Decomposition: Q_tot = Σ_k (α_k × f_k(Q_inputs)) + bias(s)
2. Shape Functions (Table 4): 3-layer MLP with ABS(weight) constraint
   - Layer 1: [ABS(LINEAR.WEIGHT), LINEAR(input, 8), ELU]
   - Layer 2: [ABS(LINEAR.WEIGHT), LINEAR(8, 4), ELU]
   - Layer 3: [ABS(LINEAR.WEIGHT), LINEAR(4, 1)]
3. Identity Semantics G_ω: VAE encoder-decoder with 32-dim hidden, generates semantic masks
4. Agent Q-Network: GRU with 64-dim hidden state
5. Attention: Credit assignment using global state s and semantics z (64-dim hidden)

Hyperparameters (from Table 3 & Appendix F.3):
- Learning rate: 0.0005 (RMSprop for Q, Adam for VAE)
- Batch size: 32
- Discount γ: 0.99
- Epsilon: 1.0 → 0.05 over 50,000 steps
- Target update: 200 steps
- VAE β: 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np
from itertools import combinations


class ShapeFunction(nn.Module):
    """
    Shape Function for GAM-based value decomposition (Table 4 in paper).
    
    Structure:
    - 1st Layer: ABS(weight) × Linear(input, 8) + ELU
    - 2nd Layer: ABS(weight) × Linear(8, 4) + ELU
    - 3rd Layer: ABS(weight) × Linear(4, 1)
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        self.elu = nn.ELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply absolute value to weights for non-negativity constraint
        x = self.elu(F.linear(x, self.fc1.weight.abs(), self.fc1.bias))
        x = self.elu(F.linear(x, self.fc2.weight.abs(), self.fc2.bias))
        x = F.linear(x, self.fc3.weight.abs(), self.fc3.bias)
        return x


class IdentitySemanticsEncoder(nn.Module):
    """
    Identity Semantics G_ω (Section 4.2, Appendix F.3).
    
    VAE-based encoder-decoder that:
    - Encodes observations to latent semantics z
    - Decodes z to reconstruct observations
    - Generates semantic masks to diagnose what agents capture
    
    Architecture: 2 FC layers with 32-dim hidden state
    Loss: MSE(recon) + β × KL (β = 0.1)
    """
    
    def __init__(self, obs_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Encoder: obs → (mean, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Decoder: z → reconstructed obs
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Mask generator: z → semantic mask
        self.mask_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.Sigmoid()
        )
    
    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        mean, log_var = h.chunk(2, dim=-1)
        return mean, log_var
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def get_mask(self, z: torch.Tensor) -> torch.Tensor:
        return self.mask_generator(z)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(obs)
        z = self.reparameterize(mean, log_var)
        recon = self.decode(z)
        mask = self.get_mask(z)
        return z, mask, recon, mean, log_var


class AgentQNetwork(nn.Module):
    """
    Individual Agent Q-Network (Appendix F.3).
    
    Architecture:
    - FC layer (obs_dim → hidden_dim)
    - GRU layer with 64-dimensional hidden state
    - ReLU activation
    - FC layer (hidden → n_actions)
    
    Optimizer: RMSprop with lr=0.0005
    """
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, rnn_hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.fc_q = nn.Linear(rnn_hidden_dim, n_actions)
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.rnn_hidden_dim)
    
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        hidden_state = self.gru(x, hidden_state)
        q_values = self.fc_q(hidden_state)
        return q_values, hidden_state


class NA2QMixer(nn.Module):
    """
    NA²Q Mixer with GAM-based Value Decomposition (Section 4, Figure 1).
    
    Q_tot = Σ_k (α_k × f_k(Q_inputs)) + bias(s)
    
    Where:
    - f_k: Shape functions (order-1 for individuals, order-2 for pairs)
    - α_k: Attention-based credits computed from state s and semantics z
    
    Order-1: n individual shape functions (f_1, ..., f_n)
    Order-2: C(n,2) pairwise shape functions (f_12, f_13, ..., f_{n-1,n})
    
    Attention mechanism uses hidden_dim=64 for w_s and w_z (Appendix F.3)
    """
    
    def __init__(self, n_agents: int, state_dim: int, latent_dim: int = 16, attention_hidden_dim: int = 64):
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Number of shape functions
        self.n_order1 = n_agents
        self.n_order2 = n_agents * (n_agents - 1) // 2
        self.n_shape_functions = self.n_order1 + self.n_order2
        
        # Pairwise indices
        self.pairwise_indices = list(combinations(range(n_agents), 2))
        
        # Order-1 shape functions: Q_i → contribution
        self.order1_shapes = nn.ModuleList([ShapeFunction(input_dim=1) for _ in range(self.n_order1)])
        
        # Order-2 shape functions: (Q_i, Q_j) → pairwise contribution
        self.order2_shapes = nn.ModuleList([ShapeFunction(input_dim=2) for _ in range(self.n_order2)])
        
        # Attention mechanism for credit assignment (hidden_dim=64)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, attention_hidden_dim), nn.ReLU())
        self.semantic_encoder = nn.Sequential(nn.Linear(latent_dim * n_agents, attention_hidden_dim), nn.ReLU())
        
        self.attention_net = nn.Sequential(
            nn.Linear(attention_hidden_dim * 2, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, self.n_shape_functions)
        )
        
        # State-dependent bias
        self.bias_net = nn.Sequential(
            nn.Linear(state_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1)
        )
    
    def forward(self, agent_q_values: torch.Tensor, state: torch.Tensor, 
                agent_semantics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = agent_q_values.size(0)
        
        # Compute shape function outputs
        shape_outputs = []
        
        # Order-1: Individual contributions
        for i in range(self.n_order1):
            q_i = agent_q_values[:, i:i+1]
            f_i = self.order1_shapes[i](q_i)
            shape_outputs.append(f_i)
        
        # Order-2: Pairwise contributions
        for idx, (i, j) in enumerate(self.pairwise_indices):
            q_i = agent_q_values[:, i:i+1]
            q_j = agent_q_values[:, j:j+1]
            q_ij = torch.cat([q_i, q_j], dim=-1)
            f_ij = self.order2_shapes[idx](q_ij)
            shape_outputs.append(f_ij)
        
        shape_outputs = torch.cat(shape_outputs, dim=-1)
        
        # Attention weights using state and semantics
        state_enc = self.state_encoder(state)
        semantics_flat = agent_semantics.view(batch_size, -1)
        semantic_enc = self.semantic_encoder(semantics_flat)
        combined = torch.cat([state_enc, semantic_enc], dim=-1)
        
        attention_logits = self.attention_net(combined)
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Weighted sum + bias
        weighted_sum = (attention_weights * shape_outputs).sum(dim=-1, keepdim=True)
        bias = self.bias_net(state)
        q_total = weighted_sum + bias
        
        return q_total, attention_weights, shape_outputs
    
    def get_individual_contributions(self, attention_weights: torch.Tensor, 
                                     shape_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get interpretable individual and pairwise contributions."""
        individual_contribs = attention_weights[:, :self.n_order1] * shape_outputs[:, :self.n_order1]
        pairwise_contribs = attention_weights[:, self.n_order1:] * shape_outputs[:, self.n_order1:]
        return individual_contribs, pairwise_contribs


class NA2Q(nn.Module):
    """
    Complete NA²Q Model.
    
    Components:
    1. Agent Q-Networks: GRU with 64-dim hidden
    2. Identity Semantics: VAE with 32-dim hidden, 16-dim latent
    3. NA²Q Mixer: GAM with order-1 and order-2 shape functions
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
        self.semantics_encoder = IdentitySemanticsEncoder(obs_dim, semantic_hidden_dim, latent_dim)
        self.mixer = NA2QMixer(n_agents, state_dim, latent_dim, attention_hidden_dim)
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.agent_q_network.init_hidden(batch_size * self.n_agents)
    
    def forward(self, observations: torch.Tensor, hidden_states: torch.Tensor,
                state: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = observations.size(0)
        
        # Process through Q-network
        obs_flat = observations.reshape(batch_size * self.n_agents, self.obs_dim)
        q_values_flat, hidden_states = self.agent_q_network(obs_flat, hidden_states)
        q_values = q_values_flat.reshape(batch_size, self.n_agents, self.n_actions)
        
        # Process through semantics encoder
        z_list, mask_list, recon_list, mean_list, logvar_list = [], [], [], [], []
        for i in range(self.n_agents):
            obs_i = observations[:, i]
            z, mask, recon, mean, logvar = self.semantics_encoder(obs_i)
            z_list.append(z)
            mask_list.append(mask)
            recon_list.append(recon)
            mean_list.append(mean)
            logvar_list.append(logvar)
        
        agent_semantics = torch.stack(z_list, dim=1)
        masks = torch.stack(mask_list, dim=1)
        recons = torch.stack(recon_list, dim=1)
        means = torch.stack(mean_list, dim=1)
        logvars = torch.stack(logvar_list, dim=1)
        
        # VAE loss: MSE + β×KL (β=0.1)
        recon_loss = F.mse_loss(recons, observations, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvars - means.pow(2) - logvars.exp())
        vae_loss = recon_loss + 0.1 * kl_loss
        
        result = {
            'q_values': q_values, 'hidden_states': hidden_states, 'masks': masks,
            'agent_semantics': agent_semantics, 'vae_loss': vae_loss
        }
        
        # Compute Q_total if state and actions provided
        if state is not None and actions is not None:
            actions_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()
            chosen_q_values = (q_values * actions_onehot).sum(dim=-1)
            q_total, attention_weights, shape_outputs = self.mixer(chosen_q_values, state, agent_semantics)
            individual_contribs, pairwise_contribs = self.mixer.get_individual_contributions(attention_weights, shape_outputs)
            
            result.update({
                'q_total': q_total, 'attention_weights': attention_weights,
                'shape_outputs': shape_outputs, 'individual_contribs': individual_contribs,
                'pairwise_contribs': pairwise_contribs
            })
        
        return result
    
    def get_q_values(self, observations: torch.Tensor, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = observations.size(0)
        obs_flat = observations.reshape(batch_size * self.n_agents, self.obs_dim)
        q_values_flat, hidden_states = self.agent_q_network(obs_flat, hidden_states)
        q_values = q_values_flat.reshape(batch_size, self.n_agents, self.n_actions)
        return q_values, hidden_states


class NA2QAgent:
    """
    NA²Q Agent for training and inference.
    
    Training settings (from paper Table 3 & Appendix F.3):
    - Learning rate: 0.0005
    - Batch size: 32
    - Discount γ: 0.99
    - Epsilon: 1.0 → 0.05 over 50,000 steps
    - Target update: 200 steps
    - VAE β: 0.1
    - Optimizers: RMSprop (Q), Adam (VAE)
    """
    
    def __init__(self, n_agents: int, obs_dim: int, state_dim: int, n_actions: int,
                 hidden_dim: int = 64, rnn_hidden_dim: int = 64, semantic_hidden_dim: int = 32,
                 latent_dim: int = 16, attention_hidden_dim: int = 64, lr: float = 5e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 50000, target_update_interval: int = 200,
                 vae_loss_weight: float = 0.1, device: str = "cpu"):
        
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.vae_loss_weight = vae_loss_weight
        self.device = torch.device(device)
        
        # Online and target networks
        self.model = NA2Q(n_agents, obs_dim, state_dim, n_actions, hidden_dim, rnn_hidden_dim,
                         semantic_hidden_dim, latent_dim, attention_hidden_dim).to(self.device)
        self.target_model = NA2Q(n_agents, obs_dim, state_dim, n_actions, hidden_dim, rnn_hidden_dim,
                                semantic_hidden_dim, latent_dim, attention_hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizers (RMSprop for Q, Adam for VAE)
        q_params = list(self.model.agent_q_network.parameters()) + list(self.model.mixer.parameters())
        vae_params = list(self.model.semantics_encoder.parameters())
        
        self.q_optimizer = torch.optim.RMSprop(q_params, lr=lr, alpha=0.99, eps=1e-5)
        self.vae_optimizer = torch.optim.Adam(vae_params, lr=lr, betas=(0.9, 0.999))
        
        # Learning rate schedulers for long training
        self.q_scheduler = torch.optim.lr_scheduler.StepLR(self.q_optimizer, step_size=5000, gamma=0.5)
        self.vae_scheduler = torch.optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=5000, gamma=0.5)
        
        self.train_step = 0
        self.hidden_states = None
    
    def init_hidden(self, batch_size: int = 1):
        self.hidden_states = self.model.init_hidden(batch_size).to(self.device)
    
    def select_actions(self, observations: np.ndarray, avail_actions: Optional[np.ndarray] = None,
                       evaluate: bool = False) -> np.ndarray:
        """Select actions using ε-greedy policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            
            if self.hidden_states is None:
                self.init_hidden(1)
            
            q_values, self.hidden_states = self.model.get_q_values(obs_tensor, self.hidden_states)
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
    
    def update_epsilon(self):
        """Update epsilon with linear decay based on training steps."""
        if self.train_step < self.epsilon_decay:
            self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * (1.0 - self.train_step / self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_end
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train_step_fn(self, batch: dict) -> dict:
        """Perform one training step. Loss = TD_loss + β × VAE_loss."""
        observations = torch.FloatTensor(batch["observations"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        next_observations = torch.FloatTensor(batch["next_observations"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        avail_actions = torch.FloatTensor(batch["avail_actions"]).to(self.device)
        
        batch_size = observations.size(0)
        seq_len = observations.size(1)
        
        hidden = self.model.init_hidden(batch_size).to(self.device)
        target_hidden = self.target_model.init_hidden(batch_size).to(self.device)
        
        q_totals, target_q_totals, vae_losses = [], [], []
        
        for t in range(seq_len):
            obs_t, state_t, actions_t = observations[:, t], states[:, t], actions[:, t]
            
            # Get done flags - dones come as [batch, seq] from replay buffer
            if dones.dim() == 2:  # [batch, seq]
                done_t = dones[:, t]
            elif dones.dim() == 1:  # [seq] - single episode
                done_t = dones[t].expand(batch_size) if batch_size > 1 else dones[t]
            else:  # scalar or unexpected
                done_t = torch.zeros(batch_size, device=self.device)
            
            # Reset hidden states for done episodes (at start of timestep)
            # This resets based on done flag from previous timestep (if t > 0)
            if t > 0:
                # Get previous timestep's done flag
                if dones.dim() == 2:
                    prev_done = dones[:, t - 1]
                elif dones.dim() == 1:
                    prev_done = dones[t - 1].expand(batch_size) if batch_size > 1 else dones[t - 1]
                else:
                    prev_done = torch.zeros(batch_size, device=self.device)
                
                done_mask = prev_done.view(batch_size, 1).expand(batch_size, hidden.size(1))
                hidden = hidden * (1 - done_mask.float())
                target_hidden = target_hidden * (1 - done_mask.float())
            
            result = self.model(obs_t, hidden, state_t, actions_t)
            q_totals.append(result['q_total'])
            vae_losses.append(result['vae_loss'])
            hidden = result['hidden_states']
            
            with torch.no_grad():
                # Use next timestep's observations and states
                if t < seq_len - 1:
                    next_obs_t = next_observations[:, t]
                    next_state_t = next_states[:, t]
                    if avail_actions.dim() == 3:
                        next_avail = avail_actions[:, t]
                    elif avail_actions.dim() == 2:
                        next_avail = avail_actions
                    else:
                        next_avail = avail_actions
                else:
                    # Last timestep: use the last next_obs/next_state
                    next_obs_t = next_observations[:, -1]
                    next_state_t = next_states[:, -1]
                    if avail_actions.dim() == 3:
                        next_avail = avail_actions[:, -1]
                    elif avail_actions.dim() == 2:
                        next_avail = avail_actions
                    else:
                        next_avail = avail_actions
                
                # Reset target hidden for done episodes
                done_mask_target = done_t.view(batch_size, 1).expand(batch_size, target_hidden.size(1))
                target_hidden = target_hidden * (1 - done_mask_target.float())
                
                target_q_values, target_hidden = self.target_model.get_q_values(next_obs_t, target_hidden)
                # Mask unavailable actions
                if next_avail.dim() == 2:
                    target_q_values[next_avail == 0] = -float('inf')
                next_actions = target_q_values.argmax(dim=-1)
                target_result = self.target_model(next_obs_t, target_hidden, next_state_t, next_actions)
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
        td_loss = F.mse_loss(q_totals, targets.detach())
        total_loss = td_loss + self.vae_loss_weight * vae_loss
        
        self.q_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.q_optimizer.step()
        self.vae_optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.update_target()
        
        # Update learning rate schedulers (for long training)
        if self.train_step % 100 == 0:  # Update every 100 steps
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




