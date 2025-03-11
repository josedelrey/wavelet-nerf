import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from nerf.encoding import positional_encoding


class NeRFModel(nn.Module):
    """
    Standard NeRF model using ReLU activations and positional encoding.
    """
    def __init__(self, pos_encoding_dim: int = 10, dir_encoding_dim: int = 4, hidden_dim: int = 256) -> None:
        super(NeRFModel, self).__init__()

        # First MLP block
        self.block1 = nn.Sequential(
            nn.Linear(pos_encoding_dim * 6 + 3, hidden_dim),  # 3D point + 3 sin/cos pairs per freq
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

        # Second MLP block
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_encoding_dim * 6 + 3, hidden_dim),  # Skip conn with original point
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # Last neuron outputs density
        )

        # RGB head
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_encoding_dim * 6 + 3, hidden_dim // 2),  # Concat with ray direction
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        self.pos_encoding_dim = pos_encoding_dim
        self.dir_encoding_dim = dir_encoding_dim

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode inputs, compute features, and output RGB color and density.
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
    """
    def __init__(self, w0: float):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenNeRFModel(nn.Module):
    """
    NeRF variant using SIREN activations.
    Processes raw 3D points and ray directions with sine-based MLPs.
    """
    def __init__(self, w0: float = 30, hidden_w0: float = 1, hidden_dim: int = 256):
        super(SirenNeRFModel, self).__init__()
        
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

        # Second MLP block
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # Skip conn with original point
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim),
            SineLayer(hidden_w0),
            nn.Linear(hidden_dim, hidden_dim + 1)  # Last neuron outputs density
        )

        # RGB head
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),  # Concat with ray direction
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
                    if i == 0:
                        bound = np.sqrt(6 / module.in_features) / hidden_w0
                        module.weight.uniform_(-bound, bound)

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode 3D points, compute features, and output RGB color and density.
        """
        features = self.block1(points)
        features = self.block2(torch.cat((features, points), dim=1))
        density = torch.relu(features[:, -1])
        features = features[:, :-1]
        colors = self.rgb_head(torch.cat((features, rays_d), dim=1))
        return colors, density
