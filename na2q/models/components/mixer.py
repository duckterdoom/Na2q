"""
NA2Q Mixer Component.
Implements the GAM-based value decomposition with shape functions and attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from itertools import combinations


class ShapeFunction(nn.Module):
    """
    Shape Function for GAM-based value decomposition.
    Implements f_k: Q_inputs → contribution.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        self.elu = nn.ELU()
        
        # Initialize weights with small positive values for stability
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


class NA2QMixer(nn.Module):
    """
    NA²Q Mixer with GAM-based Value Decomposition.
    Q_tot(s, a) = Σ_k (α_k(s, z) × f_k(Q_inputs)) + bias(s)
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
        Forward pass.
        Returns:
            q_total: [batch, 1] - Total Q-value
            attention_weights: [batch, n_shape_functions]
            shape_outputs: [batch, n_shape_functions]
        """
        batch_size = agent_q_values.size(0)
        
        # Step 1: Compute shape function outputs
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
        
        # Step 2: Compute attention weights
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
