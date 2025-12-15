"""
Agent Q-Network Component.
Implements the local Q-network for each agent using GRU memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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
