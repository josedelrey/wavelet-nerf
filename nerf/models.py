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


class SineLayer(nn.Module):
    """
    Sine activation layer with a scaling factor.

    Args:
        w0 (float): Scaling factor applied to the input before the sine activation.
    """
    def __init__(self, w0: float):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """
    NeRF variant using SIREN activations.
    Processes raw 3D points and ray directions with sine-based MLPs.

    Args:
        w0 (float): Initial scaling factor for the first SIREN activation.
        hidden_w0 (float): Scaling factor for subsequent SIREN activations.
        hidden_dim (int): Number of neurons in the hidden layers.
    """
    def __init__(self, w0: float = 30, hidden_w0: float = 1, hidden_dim: int = 256):
        super(Siren, self).__init__()
        
        # First MLP block
        self.block1 = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3D point input
            SineLayer(w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0)
        )

        # Second MLP block with skip connection
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim + 1)  # Density output
        )

        # RGB head combining features with ray direction
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        # Weight initialization for SIREN layers
        with torch.no_grad():
            # Initialize block1 linear layers
            for i, module in enumerate(self.block1):
                if isinstance(module, nn.Linear):
                    if i == 0:
                        bound = 1 / module.in_features
                    else:
                        bound = np.sqrt(6 / module.in_features) / hidden_w0
                    module.weight.uniform_(-bound, bound)
            
            # Initialize block2 linear layers
            for i, module in enumerate(self.block2):
                if isinstance(module, nn.Linear):
                    bound = np.sqrt(6 / module.in_features) / hidden_w0
                    module.weight.uniform_(-bound, bound)
            
            # Initialize RGB head first linear layer
            for i, module in enumerate(self.rgb_head):
                if isinstance(module, nn.Linear):
                    bound = np.sqrt(6 / module.in_features) / hidden_w0
                    module.weight.uniform_(-bound, bound)

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode 3D points, compute features, and output RGB color and density.

        Args:
            points (torch.Tensor): Tensor representing raw 3D points.
            rays_d (torch.Tensor): Tensor representing raw ray directions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - colors: Tensor with normalized RGB values.
                - density: Tensor representing the density values.
        """
        features = self.block1(points)
        features = self.block2(torch.cat((features, points), dim=1))
        density = torch.relu(features[:, -1])
        features = features[:, :-1]
        colors = self.rgb_head(torch.cat((features, rays_d), dim=1))
        return colors, density
