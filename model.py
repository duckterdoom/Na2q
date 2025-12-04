"""
NA²Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning

Based on ICML 2023 paper by Liu, Zhu, Chen (Nanjing University)
Paper: https://proceedings.mlr.press/v202/liu23be/liu23be.pdf
Code: https://github.com/zichuan-liu/NA2Q

Architecture (based on paper Section 4):
1. GAM-based Value Decomposition: Q_tot = Σ_k (α_k × f_k(Q_inputs)) + bias(s)
   (Implementation of the GAM framework described in Section 4)
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
    Shape Function for GAM-based value decomposition (Section 4, Table 4).
    
    Implements f_k: Q_inputs → contribution following Section 3 theoretical analysis.
    
    Architecture (Table 4):
    - Layer 1: ABS(weight) × Linear(input, 8) + ELU
    - Layer 2: ABS(weight) × Linear(8, 4) + ELU
    - Layer 3: ABS(weight) × Linear(4, 1)
    
    Theoretical Properties (Section 3):
    - ABS(weight) constraint ensures monotonicity properties
    - Enables interpretable decomposition of Q-values
    - Supports both order-1 (individual) and order-2 (pairwise) interactions
    
    Input dimensions:
    - Order-1: input_dim=1 (single Q-value Q_i)
    - Order-2: input_dim=2 (pair of Q-values [Q_i, Q_j])
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        self.elu = nn.ELU()
        
        # Initialize weights with small positive values for stable training
        # Since we use abs(weight), initializing near 0 with small variance works well
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply absolute value to weights for non-negativity constraint
        x = self.elu(F.linear(x, self.fc1.weight.abs(), self.fc1.bias))
        x = self.elu(F.linear(x, self.fc2.weight.abs(), self.fc2.bias))
        x = F.linear(x, self.fc3.weight.abs(), self.fc3.bias)
        return x


class IdentitySemanticsEncoder(nn.Module):
    """
    Identity Semantics G_ω (Figure 2, Section 4.2, Appendix F.3).
    
    VAE-based encoder-decoder following Figure 2:
    - Encoder E_ω1: h_i^t (hidden state) → z_i (identity semantic)
    - Decoder D_ω2: z_i → reconstruction (õ_i^t M_i)
    - Loss L_GW: MSE(recon) + β × KL (β = 0.1)
    
    Architecture: 2 FC layers with 32-dim hidden state
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        """
        Args:
            input_dim: Dimension of input (hidden_dim from GRU, typically 64)
            hidden_dim: VAE hidden dimension (32)
            latent_dim: Latent dimension (16)
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder E_ω1: h_i^t → (mean, log_var) (Figure 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Decoder D_ω2: z → reconstructed input (Figure 2)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Mask generator: z → semantic mask
        self.mask_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder E_ω1: h_i^t → (mean, log_var)"""
        h_enc = self.encoder(h)
        mean, log_var = h_enc.chunk(2, dim=-1)
        return mean, log_var
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder D_ω2: z_i → reconstruction"""
        return self.decoder(z)
    
    def get_mask(self, z: torch.Tensor) -> torch.Tensor:
        return self.mask_generator(z)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass following Figure 2.
        
        Args:
            h: [batch, input_dim] - Hidden state h_i^t from GRU
        
        Returns:
            z: Identity semantic z_i
            mask: Semantic mask
            recon: Reconstructed hidden state
            mean, log_var: VAE parameters
        """
        mean, log_var = self.encode(h)
        z = self.reparameterize(mean, log_var)
        recon = self.decode(z)
        mask = self.get_mask(z)
        return z, mask, recon, mean, log_var


class AgentQNetwork(nn.Module):
    """
    Individual Agent Q-Network (Figure 2, Appendix F.3).
    
    Architecture (following Figure 2):
    - Input: o_i^t (observation) and u_i^{t-1} (previous action)
    - MLP: FC layer (obs_dim + n_actions → hidden_dim)
    - GRU layer with 64-dimensional hidden state
    - MLP: FC layer (hidden → n_actions)
    - Output: Q_i(τ_i, u_i) and h_i^t
    
    Optimizer: RMSprop with lr=0.0005
    """
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, rnn_hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # First MLP: concatenate obs and previous action (Figure 2)
        self.fc1 = nn.Linear(obs_dim + n_actions, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        # Second MLP: output Q-values
        self.fc_q = nn.Linear(rnn_hidden_dim, n_actions)
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.rnn_hidden_dim)
    
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor, 
                prev_action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass following Figure 2.
        
        Args:
            obs: [batch, obs_dim] - Current observation o_i^t
            hidden_state: [batch, rnn_hidden_dim] - Previous hidden state
            prev_action: [batch, n_actions] - Previous action u_i^{t-1} (one-hot encoded)
                    If None, uses zero vector (for first step)
        """
        batch_size = obs.size(0)
        
        # Handle previous action (Figure 2: u_i^{t-1})
        if prev_action is None:
            # First step: no previous action, use zero vector
            prev_action = torch.zeros(batch_size, self.n_actions, device=obs.device)
        elif prev_action.dim() == 1:
            # Convert action index to one-hot
            prev_action = F.one_hot(prev_action.long(), num_classes=self.n_actions).float()
        
        # Concatenate observation and previous action (Figure 2)
        x = torch.cat([obs, prev_action], dim=-1)  # [batch, obs_dim + n_actions]
        x = F.relu(self.fc1(x))  # First MLP
        hidden_state = self.gru(x, hidden_state)  # GRU
        q_values = self.fc_q(hidden_state)  # Second MLP
        return q_values, hidden_state


class NA2QMixer(nn.Module):
    """
    NA²Q Mixer with GAM-based Value Decomposition (Section 4, Figure 1).
    
    Implementation of GAM-based value decomposition from Section 4:
    Q_tot(s, a) = Σ_k (α_k(s, z) × f_k(Q_inputs)) + bias(s)
    
    Note: This formula is the implementation interpretation of the GAM decomposition
    described in Section 4. The paper describes the GAM framework conceptually,
    and this is how we implement it with attention-based credit assignment.
    
    Where:
    - f_k: Shape functions (order-1 for individuals, order-2 for pairs)
      * Order-1: f_i(Q_i) for each agent i ∈ {1, ..., n}
      * Order-2: f_ij(Q_i, Q_j) for each pair (i,j) ∈ C(n,2)
    - α_k: Attention-based credits computed as:
      α_k(s, z) = softmax(Attention(s, z))_k
      where Attention(s, z) = MLP([w_s(s), w_z(z)])
      - w_s: State encoder (state_dim → 64)
      - w_z: Semantic encoder (latent_dim × n_agents → 64)
    - bias(s): State-dependent bias term
    
    Theoretical Properties (Section 3):
    - Shape functions f_k satisfy monotonicity constraints via ABS(weight)
    - Decomposition enables interpretable credit assignment
    - Attention mechanism provides dynamic credit allocation
    
    Implementation details (Appendix F.3):
    - Attention hidden dimension: 64
    - Shape function architecture: 3-layer MLP with ABS(weight) constraint
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
        """
        Forward pass implementing GAM-based value decomposition from Section 4:
        Q_tot(s, a) = Σ_k (α_k(s, z) × f_k(Q_inputs)) + bias(s)
        
        This implements the GAM framework described in Section 4 with:
        - Shape functions f_k for individual and pairwise contributions
        - Attention weights α_k for dynamic credit assignment
        - State-dependent bias term
        
        Args:
            agent_q_values: [batch, n_agents] - Q-values for chosen actions Q_i(s_i, a_i)
            state: [batch, state_dim] - Global state s
            agent_semantics: [batch, n_agents, latent_dim] - Semantic representations z
            
        Returns:
            q_total: [batch, 1] - Total Q-value Q_tot(s, a)
            attention_weights: [batch, n_shape_functions] - Credit assignment weights α_k
            shape_outputs: [batch, n_shape_functions] - Shape function outputs f_k
        """
        batch_size = agent_q_values.size(0)
        
        # Step 1: Compute shape function outputs f_k(Q_inputs)
        # Following Section 4: Order-1 (individual) and Order-2 (pairwise)
        shape_outputs = []
        
        # Order-1: f_i(Q_i) for each agent i ∈ {1, ..., n}
        # These capture individual agent contributions
        for i in range(self.n_order1):
            q_i = agent_q_values[:, i:i+1]  # Extract Q_i for agent i
            f_i = self.order1_shapes[i](q_i)  # Apply shape function f_i
            shape_outputs.append(f_i)
        
        # Order-2: f_ij(Q_i, Q_j) for each pair (i,j) ∈ C(n,2)
        # These capture pairwise interactions between agents
        for idx, (i, j) in enumerate(self.pairwise_indices):
            q_i = agent_q_values[:, i:i+1]  # Q-value for agent i
            q_j = agent_q_values[:, j:j+1]  # Q-value for agent j
            q_ij = torch.cat([q_i, q_j], dim=-1)  # Concatenate for pairwise function
            f_ij = self.order2_shapes[idx](q_ij)  # Apply shape function f_ij
            shape_outputs.append(f_ij)
        
        # Stack all shape function outputs: [batch, n_shape_functions]
        shape_outputs = torch.cat(shape_outputs, dim=-1)
        
        # Step 2: Compute attention weights α_k(s, z) = softmax(Attention(s, z))
        # Following Section 4: Attention uses both state s and semantics z
        # w_s: State encoder (Section 4, Appendix F.3)
        state_enc = self.state_encoder(state)  # [batch, 64]
        
        # w_z: Semantic encoder (Section 4, Appendix F.3)
        semantics_flat = agent_semantics.view(batch_size, -1)  # Flatten: [batch, n_agents × latent_dim]
        semantic_enc = self.semantic_encoder(semantics_flat)  # [batch, 64]
        
        # Combine state and semantic encodings
        combined = torch.cat([state_enc, semantic_enc], dim=-1)  # [batch, 128]
        
        # Compute attention logits and apply softmax
        attention_logits = self.attention_net(combined)  # [batch, n_shape_functions]
        attention_weights = F.softmax(attention_logits, dim=-1)  # [batch, n_shape_functions]
        
        # Step 3: Compute Q_tot = Σ_k (α_k × f_k) + bias(s)
        # Weighted sum of shape function outputs
        weighted_sum = (attention_weights * shape_outputs).sum(dim=-1, keepdim=True)  # [batch, 1]
        
        # State-dependent bias term
        bias = self.bias_net(state)  # [batch, 1]
        
        # Final Q_total following Section 4 formula exactly
        q_total = weighted_sum + bias  # [batch, 1]
        
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
        # Identity semantics uses hidden states (rnn_hidden_dim), not observations (Figure 2)
        self.semantics_encoder = IdentitySemanticsEncoder(rnn_hidden_dim, semantic_hidden_dim, latent_dim)
        self.mixer = NA2QMixer(n_agents, state_dim, latent_dim, attention_hidden_dim)
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.agent_q_network.init_hidden(batch_size * self.n_agents)
    
    def forward(self, observations: torch.Tensor, hidden_states: torch.Tensor,
                state: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None,
                prev_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass following Figure 2 framework.
        
        Args:
            observations: [batch, n_agents, obs_dim] - Current observations o_i^t
            hidden_states: [batch * n_agents, rnn_hidden_dim] - Previous hidden states
            state: [batch, state_dim] - Global state s
            actions: [batch, n_agents] - Current actions (for Q_tot computation)
            prev_actions: [batch, n_agents] - Previous actions u_i^{t-1} (optional, for Figure 2)
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
        z_list, mask_list, recon_list, mean_list, logvar_list = [], [], [], [], []
        for i in range(self.n_agents):
            h_i = hidden_states_reshaped[:, i]  # [batch, rnn_hidden_dim] - h_i^t from Figure 2
            z, mask, recon, mean, logvar = self.semantics_encoder(h_i)
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
        
        # VAE loss: MSE(recon, h_i^t) + β×KL (β=0.1) - reconstructing hidden states (Figure 2)
        recon_loss = F.mse_loss(recons, hidden_states_reshaped, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvars - means.pow(2) - logvars.exp())
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
    - Learning rate: 0.0005 (RMSprop for Q, Adam for VAE)
    - Batch size: 32
    - Discount γ: 0.99 (from paper Table 3)
    - Epsilon: 1.0 → 0.05 over 50,000 steps (from paper)
    - Target update: 200 steps (soft update τ=0.01 for stability)
    - VAE β: 0.1 (from paper)
    - Optimizers: RMSprop (Q), Adam (VAE)
    
    Stability Improvements:
    - Huber loss instead of MSE for TD error (robust to outliers)
    - Q-value clipping to prevent divergence
    - Soft target updates (more stable than hard copy)
    - VAE warmup period (10k steps) to let Q-learning stabilize first
    """
    
    def __init__(self, n_agents: int, obs_dim: int, state_dim: int, n_actions: int,
                 hidden_dim: int = 64, rnn_hidden_dim: int = 64, semantic_hidden_dim: int = 32,
                 latent_dim: int = 16, attention_hidden_dim: int = 64, lr: float = 5e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 50000, target_update_interval: int = 200,  # From paper: 50k steps, 200 step updates
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
        
        # Learning rate schedulers for long training (30k episodes)
        # Gentle decay: reduce by 10% every 10000 steps to maintain learning capacity
        # This helps with very long training runs without being too aggressive
        self.q_scheduler = torch.optim.lr_scheduler.StepLR(self.q_optimizer, step_size=10000, gamma=0.9)
        self.vae_scheduler = torch.optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=10000, gamma=0.9)
        
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
    
    def select_actions_batch(self, observations: np.ndarray, avail_actions: Optional[np.ndarray] = None,
                             hidden_states: Optional[torch.Tensor] = None,
                             evaluate: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Select actions for batch of environments using ε-greedy policy.
        
        Args:
            observations: [batch_size, n_agents, obs_dim]
            avail_actions: [batch_size, n_agents, n_actions] or None
            hidden_states: [batch_size * n_agents, hidden_dim] or None
            evaluate: If True, use greedy policy
            
        Returns:
            actions: [batch_size, n_agents]
            hidden_states: Updated hidden states
        """
        batch_size = observations.shape[0]
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)  # [batch, n_agents, obs_dim]
            
            if hidden_states is None:
                hidden_states = self.model.init_hidden(batch_size).to(self.device)
            
            q_values, hidden_states = self.model.get_q_values(obs_tensor, hidden_states)
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
    
    def update_target(self):
        """Hard target update (legacy - use soft_update_target instead)."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def soft_update_target(self, tau: float = 0.005):
        """
        Soft target update using Polyak averaging.
        θ_target = τ * θ_online + (1 - τ) * θ_target
        
        This provides more stable learning than periodic hard updates.
        Typical tau values: 0.001 - 0.01
        """
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    
    def train_step_fn(self, batch: dict) -> dict:
        """
        Perform one training step. Loss = TD_loss + β × VAE_loss.
        
        Key stability improvements:
        1. Huber loss instead of MSE (robust to outliers)
        2. Q-total clipping to prevent divergence
        3. Soft target update (Polyak averaging)
        4. VAE warmup period
        """
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
        n_agents = self.n_agents
        
        # Hidden states are per-agent: (batch_size * n_agents, hidden_dim)
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
            # Hidden states are (batch_size * n_agents, hidden_dim)
            # Done flags are (batch_size,), need to expand to (batch_size * n_agents, hidden_dim)
            if t > 0:
                # Get previous timestep's done flag
                if dones.dim() == 2:
                    prev_done = dones[:, t - 1]
                elif dones.dim() == 1:
                    prev_done = dones[t - 1].expand(batch_size) if batch_size > 1 else dones[t - 1]
                else:
                    prev_done = torch.zeros(batch_size, device=self.device)
                
                # Expand done flag to match hidden state shape: (batch_size * n_agents, hidden_dim)
                # Each agent in a done episode should have its hidden state reset
                prev_done_expanded = prev_done.repeat_interleave(n_agents).view(batch_size * n_agents, 1)
                done_mask = prev_done_expanded.expand(batch_size * n_agents, hidden.size(1))
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
                done_t_expanded = done_t.repeat_interleave(n_agents).view(batch_size * n_agents, 1)
                done_mask_target = done_t_expanded.expand(batch_size * n_agents, target_hidden.size(1))
                target_hidden = target_hidden * (1 - done_mask_target.float())
                
                # Double DQN: Use ONLINE network to SELECT actions, TARGET network to EVALUATE
                # This reduces overestimation bias and stabilizes training
                online_q_values, _ = self.model.get_q_values(next_obs_t, hidden.detach())
                if next_avail.dim() == 2:
                    online_q_values[next_avail == 0] = -float('inf')
                next_actions = online_q_values.argmax(dim=-1)  # Select using online network
                
                # Evaluate using target network
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
        
        # CRITICAL FIX 1: Clip Q-totals to prevent divergence
        # Even tighter clipping for better stability (reduced from ±50 to ±30)
        # This prevents unbounded Q-value growth which causes loss to increase
        q_totals_clipped = torch.clamp(q_totals, -30.0, 30.0)
        targets_clipped = torch.clamp(targets.detach(), -30.0, 30.0)
        
        # CRITICAL FIX 2: Use Huber loss (smooth L1) instead of MSE
        # Huber loss is more robust to outliers and prevents gradient explosion
        td_loss = F.smooth_l1_loss(q_totals_clipped, targets_clipped)
        
        # CRITICAL FIX 3: VAE warmup - reduce VAE influence early to let Q-learning stabilize
        # Reduced warmup: gradually increase to full weight over 10k steps (was 20k)
        # Faster VAE integration allows better semantic learning sooner
        warmup_steps = 10000
        vae_weight = self.vae_loss_weight * min(1.0, self.train_step / warmup_steps)
        total_loss = td_loss + vae_weight * vae_loss
        
        self.q_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        total_loss.backward()
        # Balanced gradient clipping for better learning (increased from 0.2 to 0.5)
        # Allows more learning capacity while still preventing gradient explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.q_optimizer.step()
        self.vae_optimizer.step()
        
        self.train_step += 1
        
        # CRITICAL FIX 4: Use soft target update (Polyak averaging) instead of hard copy
        # This provides more stable learning compared to periodic hard updates
        # Soft update: θ_target = τ * θ_online + (1 - τ) * θ_target
        # Increased tau from 0.005 to 0.01 for faster target updates and better learning
        self.soft_update_target(tau=0.01)  # Faster target update for better convergence
        
        # Update learning rate schedulers (every training step)
        # StepLR with step_size=5000 means LR decays every 5000 calls to .step()
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




