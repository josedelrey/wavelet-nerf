import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from modules.encoding import positional_encoding


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
    SIREN-ized NeRF model.
    
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


class WaveletFilter(nn.Module):
    """
    A Morlet wavelet filter module used for feature extraction.

    This layer applies a set of learnable Morlet wavelet filters to the input tensor.
    The filters are parameterized by learned means (mu) and gamma values, similar to
    the Gabor filter. Instead of using a sine nonlinearity, it uses a Morlet wavelet
    function:
    
        ψ(u) = -e̶x̶p̶(̶-̶u̶²̶/̶2̶) * cos(ω₀ * u) - exp(-ω₀²/2)
    
    where ω₀ is a learnable frequency parameter.
    
    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        alpha (float): A scaling factor for the gamma distribution.
        beta (float, optional): The rate parameter for the Gamma distribution.
        omega0 (float): Initial frequency parameter for the Morlet wavelet.
    """
    def __init__(self, in_dim: int, out_dim: int, alpha: float, beta: float = 1.0, omega0: float = 5.0) -> None:
        super(WaveletFilter, self).__init__()
        # Learned centers for each filter
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        # Learned gamma values controlling the Gaussian envelope width
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim,)))
        # Linear projection to generate the argument for the wavelet nonlinearity
        self.linear = nn.Linear(in_dim, out_dim)
        # Learnable frequency parameter for the Morlet wavelet
        self.omega0 = nn.Parameter(torch.tensor(omega0))
        self.init_weights()
    
    def init_weights(self) -> None:
        # Scale the weights based on gamma (similar to GaborFilter)
        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)
    
    def morlet_wavelet(self, u: torch.Tensor) -> torch.Tensor:
        """
        Applies the Morlet wavelet nonlinearity to the input u.
        
        ψ(u) = -e̶x̶p̶(̶-̶u̶²̶/̶2̶) * cos(ω₀ * u) - exp(-ω₀²/2)
        """
        return torch.cos(self.omega0 * u) - torch.exp(-0.5 * (self.omega0**2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute squared Euclidean distance between x and each filter's center
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        # Gaussian envelope based on the learned gamma values
        envelope = torch.exp(- self.gamma.unsqueeze(0) / 2. * norm)
        # Linear projection of x
        lin_out = self.linear(x)
        # Apply the Morlet wavelet nonlinearity
        wavelet_response = self.morlet_wavelet(lin_out)
        # Return the modulated response
        return envelope * wavelet_response


class WaveletMFN(nn.Module):
    """
    A network based on multiple Morlet wavelet filters for feature extraction and transformation.

    This model mirrors the structure of GaborNet but replaces the Gabor filters with learnable
    Morlet wavelet filters. For an input coordinate (e.g., (x, y, z)), the network first extracts
    a high-dimensional feature vector using a MorletWaveletFilter. In subsequent layers, it applies
    a linear transformation whose output is element-wise multiplied by a fresh Morlet filter response
    (computed from the same coordinate). Finally, a linear layer decodes the high-dimensional features
    into the output (e.g., density or an intermediate feature embedding).

    Args:
        in_features (int): Number of input features (e.g., 3 for spatial coordinates).
        hidden_features (int): Number of features in the hidden layers.
        out_features (int): Number of output features.
        hidden_layers (int): Number of hidden layers in the network.
        omega0 (float): Initial frequency parameter for the Morlet wavelet.
    """
    def __init__(self, 
                 in_features: int = 2, 
                 hidden_features: int = 256, 
                 out_features: int = 1, 
                 hidden_layers: int = 4,
                 omega0: float = 5.0) -> None:
        super(WaveletMFN, self).__init__()
        self.hidden_layers = hidden_layers

        # Initialize a list of MorletWaveletFilter modules (one per layer)
        self.morlet_filters = nn.ModuleList([
            WaveletFilter(in_features, hidden_features, alpha=6.0 / hidden_layers, omega0=omega0)
            for _ in range(hidden_layers)
        ])

        # Initialize the linear layers.
        # For hidden_layers - 1 layers, we have a linear mapping from hidden_features to hidden_features,
        # and the final layer maps from hidden_features to out_features.
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features) for _ in range(hidden_layers - 1)] +
            [nn.Linear(hidden_features, out_features)]
        )

        # Initialize weights for the linear layers for hidden layers
        for lin in self.linear[:hidden_layers - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_features), np.sqrt(1.0 / hidden_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First wavelet filter: get initial high-dimensional feature vector from the input coordinate
        z = self.morlet_filters[0](x)
        # Recursively apply linear layers and modulate with subsequent Morlet filter responses
        for i in range(self.hidden_layers - 1):
            z = self.linear[i](z) * self.morlet_filters[i + 1](x)
        # Final linear transformation to decode into the output
        return self.linear[self.hidden_layers - 1](z)


class WaveletMFNNeRF(nn.Module):
    """
    A NeRF model that uses two separate WaveletMFN networks:
    1. A density branch that predicts density based solely on spatial coordinates.
    2. A color branch that computes spatial features and, after fusing them with a positional encoding
       of the ray direction, produces an RGB color.
    
    The density is scaled and passed through ReLU, while the color branch applies an RGB head
    similar to the Siren architecture.
    
    Args:
        hidden_features (int): Number of hidden features for both branches.
        density_hidden_layers (int): Number of hidden layers for the density branch.
        color_hidden_layers (int): Number of hidden layers for the color feature branch.
        dir_encoding_dim (int): Number of frequencies for the ray direction positional encoding.
        sigma_mul (float): Multiplicative factor for the density output.
        rgb_mul (float): Multiplicative factor applied before color activation.
        omega0 (float): Frequency parameter used in both WaveletMFN networks.
    """
    def __init__(self,
                 hidden_features: int = 256,
                 density_hidden_layers: int = 4,
                 color_hidden_layers: int = 4,
                 dir_encoding_dim: int = 4,
                 sigma_mul: float = 10.0,
                 rgb_mul: float = 1.0,
                 omega0: float = 5.0) -> None:
        super(WaveletMFNNeRF, self).__init__()
        self.dir_encoding_dim = dir_encoding_dim
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul

        # Density branch: computes density from spatial coordinates (input: 3D point; output: 1 scalar)
        self.density_mfn = WaveletMFN(
            in_features=3,
            hidden_features=hidden_features,
            out_features=1,
            hidden_layers=density_hidden_layers,
            omega0=omega0
        )

        # Color branch: computes an intermediate feature embedding from spatial coordinates.
        # The output dimension is equal to hidden_features so that it can be fused with encoded ray directions.
        self.color_mfn = WaveletMFN(
            in_features=3,
            hidden_features=hidden_features,
            out_features=hidden_features,
            hidden_layers=color_hidden_layers,
            omega0=omega0
        )

        # RGB head: combines the feature from color_mfn with the positional encoding of the ray direction.
        # The ray direction is encoded with a size of 6 * dir_encoding_dim + 3.
        ray_encoding_size = 6 * dir_encoding_dim + 3
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_features + ray_encoding_size, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 3)
        )

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute density from the density branch: shape (batch, 1)
        density_raw = self.density_mfn(points)
        density = torch.relu(density_raw) * self.sigma_mul

        # Compute spatial features from the color branch: shape (batch, hidden_features)
        features = self.color_mfn(points)
        # Encode the viewing (ray) directions using the provided positional encoding.
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)
        # Concatenate the spatial features with the encoded direction information.
        rgb_input = torch.cat((features, rays_d_enc), dim=-1)
        # Process through the RGB head and apply scaling followed by a sigmoid to constrain to [0, 1].
        rgb = torch.sigmoid(self.rgb_head(rgb_input) * self.rgb_mul)

        return rgb, density.squeeze(-1)
