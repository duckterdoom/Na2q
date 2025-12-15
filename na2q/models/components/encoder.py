"""
Identity Semantics Encoder Component.
VAE-based encoder for generating agent semantic representations.
"""

import torch
import torch.nn as nn
from typing import Tuple


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
