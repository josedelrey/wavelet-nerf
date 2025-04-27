import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.nn import functional as F

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
    3
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


class MultiScaleWaveletFilter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, alpha: float, beta: float = 1.0,
                 low_omega0: float = 2.0, high_omega0: float = 10.0) -> None:
        """
        Splits filters into two groups:
          - One group uses a low ω₀ for low-frequency (broader Gaussian) responses.
          - The other group uses a high ω₀ for high-frequency (narrower Gaussian) responses.
        The outputs of both groups are concatenated along the feature dimension.
        
        Args:
            in_dim: Dimensionality of the input.
            out_dim: Total number of filters.
            alpha, beta: Parameters for the gamma distribution.
            low_omega0: Frequency parameter for the low-frequency group.
            high_omega0: Frequency parameter for the high-frequency group.
        """
        super(MultiScaleWaveletFilter, self).__init__()
        # Split filters into low and high frequency groups.
        low_dim = int(0.2 * out_dim)
        high_dim = out_dim - low_dim

        self.low_wavelet = WaveletFilter(in_dim, low_dim, alpha, beta, omega0=low_omega0)
        self.high_wavelet = WaveletFilter(in_dim, high_dim, alpha, beta, omega0=high_omega0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_response = self.low_wavelet(x)
        high_response = self.high_wavelet(x)
        # Concatenate features along the last dimension.
        return torch.cat([low_response, high_response], dim=-1)


class MultiScaleWaveletNeRF(nn.Module):
    """
    A NeRF backbone where each multi-scale Morlet wavelet filter
    receives the raw 3D point coordinates, with additive residual
    skips to preserve high-frequency detail.
    """
    def __init__(
        self,
        point_dim: int = 3,
        dir_dim: int = 3,
        num_freqs_dir: int = 4,
        hidden_features: int = 256,
        hidden_layers: int = 9,
        alpha: float = 0.05,
        beta: float = 0.025,
        low_omega0: float = 5.0,
        high_omega0: float = 7.0
    ):
        super().__init__()

        self.num_freqs_dir = num_freqs_dir
        self.hidden_layers = hidden_layers

        # 1. Wavelet filters
        self.wavelet_filters = nn.ModuleList([
            MultiScaleWaveletFilter(point_dim,
                                    hidden_features,
                                    alpha, beta,
                                    low_omega0, high_omega0)
            for _ in range(hidden_layers)
        ])

        # 2. Linear transforms for the hidden feature vector
        self.modulation_linears = nn.ModuleList([
            nn.Linear(hidden_features, hidden_features)
            for _ in range(hidden_layers - 1)
        ])
        for lin in self.modulation_linears:
            lin.weight.data.uniform_(
                -np.sqrt(1.0 / hidden_features),
                 np.sqrt(1.0 / hidden_features)
            )

        # Additive skip scales for each layer
        self.skip_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0))
            for _ in range(hidden_layers - 1)
        ])

        # 3. Density (σ) head
        self.density_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 1)
        )

        # 4. Colour (RGB) head
        dir_enc_dim   = dir_dim + 2 * dir_dim * num_freqs_dir
        colour_in_dim = hidden_features + dir_enc_dim
        self.color_head = nn.Sequential(
            nn.Linear(colour_in_dim, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 3)
        )

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor):
        """
        Args:
            points : (N, 3)     3-D points along the ray
            rays_d : (N, 3)     corresponding ray directions
        Returns:
            rgb     : (N, 3)    per-point colour
            density : (N,)      per-point volume density σ
        """
        # First layer: direct wavelet expansion of the coordinates
        z = self.wavelet_filters[0](points)

        # Hidden layers with multiplicative modulation + additive skip
        for i in range(1, self.hidden_layers):
            w_i = self.wavelet_filters[i](points)            # (N, D)
            h_i = self.modulation_linears[i - 1](z)          # (N, D)
            # Modulate and then add residual skip
            z = h_i * w_i + self.skip_scales[i - 1] * w_i   # (N, D)

        # Density head (non-negative with relu)
        density = torch.relu(self.density_head(z))

        # Directional encoding → colour head
        rays_d_enc = positional_encoding(rays_d, self.num_freqs_dir)
        rgb_input  = torch.cat([z, rays_d_enc], dim=-1)
        rgb        = torch.sigmoid(self.color_head(rgb_input))

        return rgb, density.squeeze(-1)
