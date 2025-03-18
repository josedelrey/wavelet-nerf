import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from nerf.encoding import positional_encoding


class NeRF(nn.Module):
    """
    Standard NeRF model using ReLU activations and positional encoding.

    Args:
        pos_encoding_dim (int): Number of frequencies for 3D point encoding.
        dir_encoding_dim (int): Number of frequencies for ray direction encoding.
        hidden_dim (int): Number of neurons in hidden layers.
    """
    def __init__(self,
                 pos_encoding_dim: int = 10,
                 dir_encoding_dim: int = 4,
                 hidden_dim: int = 256) -> None:
        super().__init__()

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

    def forward(self,
                points: torch.Tensor,
                rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        points_enc = positional_encoding(points, self.pos_encoding_dim)
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)

        features = self.block1(points_enc)
        features = self.block2(torch.cat((features, points_enc), dim=1))

        density = torch.relu(features[:, -1])
        features = features[:, :-1]
        colors = self.rgb_head(torch.cat((features, rays_d_enc), dim=1))

        return colors, density


class Sine(nn.Module):
    """
    Sine activation module.

    Args:
        w0 (float): Frequency scaling factor for the sine activation.
    """
    def __init__(self, w0: float = 30.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """
    SIREN layer consisting of a linear transformation followed by a sine activation.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the output features.
        w0 (float): Frequency scaling factor for the sine activation.
        is_first (bool): If True, applies a different weight initialization.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 w0: float = 1.0,
                 is_first: bool = False) -> None:
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (np.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        num_layers (int): Number of layers in the base MLP.
        hidden_dim (int): Hidden dimension.
        dir_encoding_dim (int): Number of frequencies for ray direction encoding.
        sigma_mul (float): Multiplicative factor for density output.
        rgb_mul (float): Multiplicative factor for RGB output.
        w0 (float): w0 parameter for the first SIREN layer.
        hidden_w0 (float): w0 parameter for subsequent SIREN layers.
    """
    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 256,
        dir_encoding_dim: int = 4,
        sigma_mul: float = 10.,
        rgb_mul: float = 1.,
        w0: float = 30.,
        hidden_w0: float = 1.)-> None:
        super().__init__()
        self.dir_encoding_dim = dir_encoding_dim
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul

        # Base MLP: 3D point processing
        base_layers = [SirenLayer(3, hidden_dim, w0=w0, is_first=True)]
        for _ in range(num_layers - 1):
            base_layers.append(SirenLayer(hidden_dim, hidden_dim, w0=hidden_w0))
        self.block1 = nn.Sequential(*base_layers)

        # Density branch: outputs density from base features
        self.density_branch = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

        # Feature remapping: prepares features for the RGB head
        self.feature_remap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )

        # RGB head: combines remapped features with encoded ray directions
        ray_encoding_size = 6 * self.dir_encoding_dim + 3
        self.rgb_head = nn.Sequential(
            SirenLayer(hidden_dim + ray_encoding_size, hidden_dim // 2, w0=hidden_w0),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, 
                points: torch.Tensor, 
                rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process points through the base MLP
        base = self.block1(points)

        # Compute density from base features
        sigma = self.density_branch(base)
        density = torch.relu(sigma) * self.sigma_mul

        # Remap features and encode ray directions for RGB head
        features = self.feature_remap(base)
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)
        rgb_input = torch.cat((features, rays_d_enc), dim=-1)
        rgb = self.rgb_head(rgb_input)

        # Scale and constrain rgb values to [0, 1]
        rgb = torch.sigmoid_(rgb * self.rgb_mul)
        return rgb, density.squeeze(-1)
