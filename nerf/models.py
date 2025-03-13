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

        # First MLP block
        self.block1 = nn.Sequential(
            nn.Linear(pos_encoding_dim * 6 + 3, hidden_dim),  # 3D point + sin/cos pairs per frequency
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
            nn.Linear(hidden_dim, hidden_dim + 1)  # Last neuron outputs density
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
    Linear layer with sine activation.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): If True, uses bias.
        is_first (bool): If True, this is the first layer.
        omega_0 (float): Frequency scaling factor.
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        """
        SIREN-specific weight initialization.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation and sine activation.
        """
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the sine-activated output and the pre-activation value.
        """
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    """
    SIREN network using sine activations.
    
    Processes concatenated 3D points and ray directions to output RGB and density.
    
    Args:
        in_features (int): Input dimension (typically 6).
        hidden_features (int): Neurons in hidden layers.
        hidden_layers (int): Number of hidden layers.
        out_features (int): Output dimension (typically 4: 3 for RGB, 1 for density).
        outermost_linear (bool): If True, use a final linear layer.
        first_omega_0 (float): Omega_0 for the first layer.
        hidden_omega_0 (float): Omega_0 for hidden layers.
    """
    def __init__(self, in_features: int = 6, hidden_features: int = 256, hidden_layers: int = 6, 
                 out_features: int = 4, outermost_linear: bool = True, 
                 first_omega_0: float = 30, hidden_omega_0: float = 1):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0, 
                    np.sqrt(6 / hidden_features) / hidden_omega_0
                )
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: concatenates inputs, processes through the network, and splits output into RGB and density.
        """
        inputs = torch.cat((points, rays_d), dim=1)
        output = self.net(inputs)
        colors = torch.sigmoid(output[:, :3])
        density = torch.relu(output[:, 3])
        return colors, density
