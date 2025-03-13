import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from nerf.encoding import positional_encoding
from collections import OrderedDict


class NeRF(nn.Module):
    """
    Standard NeRF model using ReLU activations and positional encoding.
    
    Args:
        pos_encoding_dim (int): Number of frequencies for positional encoding of 3D points.
        dir_encoding_dim (int): Number of frequencies for positional encoding of ray directions.
        hidden_dim (int): Number of neurons in the hidden layers.
    """
    def __init__(self, pos_encoding_dim: int = 10, dir_encoding_dim: int = 4, hidden_dim: int = 256) -> None:
        super(NeRF, self).__init__()

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

        Args:
            points (torch.Tensor): Tensor representing input 3D points.
            rays_d (torch.Tensor): Tensor representing input ray directions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - colors: Tensor with normalized RGB values.
                - density: Tensor representing the density values.
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
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
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
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate the 3D points and ray directions (each assumed to be of shape [B, 3]) into a [B, 6] input.
        inputs = torch.cat((points, rays_d), dim=1)
        output = self.net(inputs)
        
        # Split the output: first 3 channels are RGB (apply sigmoid) and the last channel is density (apply ReLU).
        colors = torch.sigmoid(output[:, :3])
        density = torch.relu(output[:, 3])
        return colors, density
