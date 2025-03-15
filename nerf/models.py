import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from nerf.encoding import positional_encoding


class NeRF(nn.Module):
    """
    Standard NeRF model using ReLU activations and positional encoding.
    
    Args:
        pos_encoding_dim (int): Number of frequencies for 3D point encoding.
        dir_encoding_dim (int): Number of frequencies for ray direction encoding.
        hidden_dim (int): Number of neurons in hidden layers.
    """
    def __init__(self, pos_encoding_dim: int = 10, dir_encoding_dim: int = 4, hidden_dim: int = 256) -> None:
        super(NeRF, self).__init__()

        # First MLP block (input: 3D point + sin/cos pairs per frequency)
        self.block1 = nn.Sequential(
            nn.Linear(pos_encoding_dim * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Second MLP block with skip connection
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_encoding_dim * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # Density output
        )

        # RGB head combining features with ray direction
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_encoding_dim * 6 + 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        self.pos_encoding_dim = pos_encoding_dim
        self.dir_encoding_dim = dir_encoding_dim

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encodes inputs and outputs RGB and density.
        """
        points_enc = positional_encoding(points, self.pos_encoding_dim)
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)
        features = self.block1(points_enc)
        features = self.block2(torch.cat((features, points_enc), dim=1))
        density = torch.relu(features[:, -1])
        features = features[:, :-1]
        colors = self.rgb_head(torch.cat((features, rays_d_enc), dim=1))
        return colors, density


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1., is_first=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (np.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            if self.layer.bias is not None:
                self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.layer(x) 
        out = self.activation(out)
        return out


class Siren(nn.Module):
    """
    SIREN-ized NeRF model with a structure similar to the NeRF class.
    
    This model always uses separate branches for density and appearance (RGB).
    It processes input 3D points with a base MLP (using SIREN layers),
    then splits the computation into a density branch and an RGB head.
    The RGB head combines remapped features with a positional encoding of ray directions.
    
    Args:
        D (int): Number of layers in the base MLP.
        W (int): Hidden dimension.
        dir_encoding_dim (int): Number of frequencies for ray direction encoding.
        sigma_mul (float): Multiplicative factor for density output.
        rgb_mul (float): Multiplicative factor for RGB output.
        first_layer_w0 (float): w0 parameter for the first SIREN layer.
        following_layers_w0 (float): w0 parameter for subsequent SIREN layers.
    """
    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        dir_encoding_dim: int = 4,
        sigma_mul: float = 10.,
        rgb_mul: float = 1.,
        first_layer_w0: float = 30.,
        following_layers_w0: float = 1.,
        **not_used_kwargs
    ):
        super().__init__()
        use_bias = True
        self.dir_encoding_dim = dir_encoding_dim
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul

        # -------------------------
        # Block 1: Base MLP for 3D point processing
        # -------------------------
        # Input: 3D points.
        base_layers = [SirenLayer(3, W, use_bias=use_bias, w0=first_layer_w0, is_first=True)]
        for _ in range(D - 1):
            base_layers.append(SirenLayer(W, W, use_bias=use_bias, w0=following_layers_w0))
        self.block1 = nn.Sequential(*base_layers)

        # -------------------------
        # Density Branch: outputs density from base features
        # -------------------------
        self.density_branch = nn.Sequential(
            nn.Linear(W, 1, bias=use_bias)
        )
        # -------------------------
        # Feature Remapping: prepares features for the RGB head
        # -------------------------
        self.feature_remap = nn.Sequential(
            nn.Linear(W, W, bias=use_bias)
        )
        # -------------------------
        # RGB Head: combines remapped features with encoded ray directions
        # -------------------------
        # Compute ray encoding size: (6 * dir_encoding_dim + 3)
        ray_encoding_size = self.dir_encoding_dim * 6 + 3
        self.rgb_head = nn.Sequential(
            SirenLayer(W + ray_encoding_size, W // 2, use_bias=use_bias, w0=following_layers_w0),
            nn.Linear(W // 2, 3, bias=use_bias)
        )

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: processes 3D points and ray directions and returns RGB colors and density.
        
        Args:
            points (torch.Tensor): Input 3D points.
            rays_d (torch.Tensor): Input ray directions.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (rgb, density)
        """
        # Process points through the base MLP.
        base = self.block1(points)

        # Compute density from base features.
        sigma = self.density_branch(base)
        density = torch.relu(sigma) * self.sigma_mul

        # Remap base features before RGB prediction.
        features = self.feature_remap(base)
        # Encode ray directions using positional encoding.
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)
        # Concatenate remapped features with encoded ray directions.
        rgb_input = torch.cat((features, rays_d_enc), dim=-1)
        rgb = self.rgb_head(rgb_input)

        # Scale and constrain rgb values to [0, 1].
        rgb = torch.sigmoid_(rgb * self.rgb_mul)
        return rgb, density.squeeze(-1)
