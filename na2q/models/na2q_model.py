"""
NA²Q Model - Core Neural Network Architecture.

Implements the NA²Q model combining:
- Agent Q-Networks (GRU-based)
- Identity Semantics Encoder (VAE)
- NA²Q Mixer (GAM-based value decomposition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .components.network import AgentQNetwork
from .components.encoder import IdentitySemanticsEncoder
from .components.mixer import NA2QMixer


# =============================================================================
# NA²Q Model
# =============================================================================

class NA2Q(nn.Module):
    """
    Complete NA²Q Model.
    
    Architecture (Figure 2 from paper):
    - Agent Q-Network: o_i^t, u_i^{t-1} → MLP → GRU → MLP → Q_i, h_i^t
    - Semantics Encoder: h_i^t → VAE → z_i
    - Mixer: Q_i values + z_i + state → Q_total (GAM)
    """
    
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
        attention_hidden_dim: int = 64
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        
        # Components
        self.agent_q_network = AgentQNetwork(obs_dim, n_actions, hidden_dim, rnn_hidden_dim)
        self.semantics_encoder = IdentitySemanticsEncoder(rnn_hidden_dim, semantic_hidden_dim, latent_dim)
        self.mixer = NA2QMixer(n_agents, state_dim, latent_dim, attention_hidden_dim)
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.agent_q_network.init_hidden(batch_size * self.n_agents)
    
    def forward(
        self,
        observations: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
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
        
        # Q-Network forward
        obs_flat = observations.reshape(batch_size * self.n_agents, self.obs_dim)
        prev_actions_flat = None
        if prev_actions is not None:
            prev_actions_flat = prev_actions.reshape(batch_size * self.n_agents)
        
        q_values_flat, hidden_states = self.agent_q_network(obs_flat, hidden_states, prev_actions_flat)
        q_values = q_values_flat.reshape(batch_size, self.n_agents, self.n_actions)
        
        # Semantics Encoder (VAE)
        hidden_reshaped = hidden_states.view(batch_size, self.n_agents, -1)
        hidden_flat = hidden_reshaped.reshape(batch_size * self.n_agents, -1)
        z_flat, mask_flat, recon_flat, mean_flat, logvar_flat = self.semantics_encoder(hidden_flat)
        
        agent_semantics = z_flat.view(batch_size, self.n_agents, -1)
        masks = mask_flat.view(batch_size, self.n_agents, -1)
        
        # VAE Loss: MSE(recon, h) + β×KL
        recon_loss = F.mse_loss(recon_flat, hidden_flat, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar_flat - mean_flat.pow(2) - logvar_flat.exp())
        vae_loss = recon_loss + 0.1 * kl_loss
        
        result = {
            'q_values': q_values,
            'hidden_states': hidden_states,
            'masks': masks,
            'agent_semantics': agent_semantics,
            'vae_loss': vae_loss
        }
        
        # Compute Q_total if state and actions provided
        if state is not None and actions is not None:
            actions_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()
            chosen_q_values = (q_values * actions_onehot).sum(dim=-1)
            
            q_total, attention_weights, shape_outputs = self.mixer(chosen_q_values, state, agent_semantics)
            individual_contribs, pairwise_contribs = self.mixer.get_individual_contributions(
                attention_weights, shape_outputs
            )
            
            result.update({
                'q_total': q_total,
                'attention_weights': attention_weights,
                'shape_outputs': shape_outputs,
                'individual_contribs': individual_contribs,
                'pairwise_contribs': pairwise_contribs
            })
        
        return result
    
    def get_q_values(
        self,
        observations: torch.Tensor,
        hidden_states: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values without mixer (for action selection)."""
        batch_size = observations.size(0)
        obs_flat = observations.reshape(batch_size * self.n_agents, self.obs_dim)
        
        prev_actions_flat = None
        if prev_actions is not None:
            prev_actions_flat = prev_actions.reshape(batch_size * self.n_agents)
        
        q_values_flat, hidden_states = self.agent_q_network(obs_flat, hidden_states, prev_actions_flat)
        q_values = q_values_flat.reshape(batch_size, self.n_agents, self.n_actions)
        return q_values, hidden_states
