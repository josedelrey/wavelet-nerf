import torch
import torch.nn as nn
from typing import Tuple
from nerf.encoding import positional_encoding


class NeRFModel(nn.Module):
    def __init__(self, pos_encoding_dim: int = 10, dir_encoding_dim: int = 4, hidden_dim: int = 256) -> None:
        """
        Initializes the NeRF model with two MLP blocks and an RGB head.

        Args:
            pos_encoding_dim (int): Positional encoding length for 3D points.
            dir_encoding_dim (int): Positional encoding length for ray directions.
            hidden_dim (int): Number of hidden units in the hidden layers.
        """
        super(NeRFModel, self).__init__()

        # First MLP block for processing positional-encoded points.
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

        # Second MLP block with a skip connection.
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_encoding_dim * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )

        # RGB prediction head.
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
        Forward pass: applies positional encoding and processes through MLPs
        to obtain color and density predictions.

        Args:
            points (torch.Tensor): Input 3D points (shape: [N, 3]).
            rays_d (torch.Tensor): Input ray directions (shape: [N, 3]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted colors (RGB, shape: [N, 3])
            and density (shape: [N, 1]).
        """
        # Apply positional encoding to points and ray directions.
        points_enc = positional_encoding(points, self.pos_encoding_dim)
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)

        # Process through the first MLP block.
        features = self.block1(points_enc)
        
        # Concatenate skip connection and process through the second MLP block.
        features = self.block2(torch.cat((features, points_enc), dim=1))
        density = torch.relu(features[:, -1])  # Density prediction
        features = features[:, :-1]

        # Predict RGB values using the RGB head.
        colors = self.rgb_head(torch.cat((features, rays_d_enc), dim=1))
        return colors, density
