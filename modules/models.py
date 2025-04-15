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
        low_dim = out_dim // 2
        high_dim = out_dim - low_dim

        self.low_wavelet = WaveletFilter(in_dim, low_dim, alpha, beta, omega0=low_omega0)
        self.high_wavelet = WaveletFilter(in_dim, high_dim, alpha, beta, omega0=high_omega0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_response = self.low_wavelet(x)
        high_response = self.high_wavelet(x)
        # Concatenate features along the last dimension.
        return torch.cat([low_response, high_response], dim=-1)
    

class WaveletFilterLearnable(nn.Module):
    """
    A wavelet filter module with learnable alpha and beta parameters.

    Instead of fixing alpha and beta for sampling a Gamma-distributed parameter,
    this version learns per-filter alpha and beta, and uses them to compute 
    gamma = softplus(alpha) / softplus(beta) as the effective envelope width.
    """
    def __init__(self, in_dim: int, out_dim: int, 
                 init_alpha: float = 2.0, init_beta: float = 1.0, 
                 omega0: float = 5.0) -> None:
        """
        Args:
            in_dim (int): Input feature dimensionality.
            out_dim (int): Number of filters / output dimensionality.
            init_alpha (float): Initial value for alpha (shape parameter).
            init_beta (float): Initial value for beta (rate parameter).
            omega0 (float): Initial frequency parameter for the Morlet wavelet.
        """
        super(WaveletFilterLearnable, self).__init__()
        
        # Initialize learnable alpha and beta as vectors (one per filter)
        # We wrap them with nn.Parameter and later use softplus to ensure positivity.
        self.alpha = nn.Parameter(torch.full((out_dim,), init_alpha))
        self.beta = nn.Parameter(torch.full((out_dim,), init_beta))
        
        # Instead of sampling gamma once, we compute it in forward pass:
        #   gamma = softplus(alpha) / softplus(beta)
        # This value will change as alpha and beta are updated.
        
        # Learned centers for each filter: shape (out_dim, in_dim)
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        
        # Linear projection layer
        self.linear = nn.Linear(in_dim, out_dim)
        
        # Learnable frequency parameter for the Morlet wavelet.
        # (We keep omega0 as a learnable parameter too, if desired.)
        self.omega0 = nn.Parameter(torch.tensor(omega0))
        
        self.init_weights()
    
    def init_weights(self) -> None:
        # We initialize the linear weights scaled by a factor based on gamma.
        # Note: Since gamma will be computed in the forward pass, we use the initial ratio.
        init_gamma = F.softplus(self.alpha).detach() / (F.softplus(self.beta).detach() + 1e-6)
        self.linear.weight.data *= 128. * torch.sqrt(init_gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)
    
    def morlet_wavelet(self, u: torch.Tensor) -> torch.Tensor:
        # Apply the Morlet wavelet nonlinearity.
        return torch.cos(self.omega0 * u) - torch.exp(-0.5 * (self.omega0 ** 2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute gamma from the learnable alpha and beta.
        # The softplus ensures gamma is positive.
        gamma = F.softplus(self.alpha) / (F.softplus(self.beta) + 1e-6)
        
        # Compute squared Euclidean distances between x and each filter's center.
        # Shape: (batch_size, out_dim)
        norm = (x ** 2).sum(dim=1, keepdim=True) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        
        # Compute the Gaussian envelope using the computed gamma.
        envelope = torch.exp(- gamma.unsqueeze(0) / 2. * norm)
        
        # Apply the linear projection and the wavelet nonlinearity.
        lin_out = self.linear(x)
        wavelet_response = self.morlet_wavelet(lin_out)
        
        # Return the modulated response.
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


class WaveletMFNNeRFSimple(nn.Module):
    """
    A simplified NeRF model based on a WaveletMFN backbone.
    This model takes 3D coordinates as input and directly predicts both
    the density (volume occupancy) and the RGB color for each point.
    
    Args:
        in_features (int): Number of input features (default: 3 for (x,y,z)).
        hidden_features (int): Number of features in the hidden layers.
        hidden_layers (int): Number of hidden layers in the WaveletMFN backbone.
        omega0 (float): Initial frequency parameter for the Morlet wavelets.
        sigma_mul (float): Multiplicative factor applied to the density output.
        rgb_mul (float): Multiplicative factor applied to the RGB output.
    """
    def __init__(self, 
                 in_features: int = 3,
                 hidden_features: int = 512,
                 hidden_layers: int = 4,
                 omega0: float = 5.0,
                 sigma_mul: float = 1.0,
                 rgb_mul: float = 1.0) -> None:
        super().__init__()
        # WaveletMFN backbone produces a feature vector from the input coordinates.
        # Here, we set out_features equal to hidden_features so that we can feed
        # the resulting features to separate heads.
        self.backbone = WaveletMFN(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=hidden_features,
            hidden_layers=hidden_layers,
            omega0=omega0
        )
        
        # Density head: outputs a single scalar value (density) from features.
        self.density_head = nn.Linear(hidden_features, 1)
        
        # RGB head: transforms features into an RGB color vector.
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 3),
            nn.Sigmoid()  # Ensures values are in the [0, 1] range.
        )
        
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            points (torch.Tensor): Input 3D coordinates, shape (N, 3).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - rgb: Predicted RGB colors, shape (N, 3).
                - density: Predicted densities, shape (N,).
        """
        # Extract features using the WaveletMFN backbone.
        features = self.backbone(points)
        
        # Predict density. Apply ReLU to enforce non-negativity, then scale.
        sigma = self.density_head(features)
        density = torch.relu(sigma) * self.sigma_mul
        
        # Predict RGB colors. The RGB head includes a Sigmoid, and we further scale the input.
        rgb = self.rgb_head(features)
        rgb = torch.sigmoid(rgb * self.rgb_mul)
        
        return rgb, density.squeeze(-1)


class WaveletMFNNeRFSeparate(nn.Module):
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
                 sigma_mul: float = 1.0,
                 rgb_mul: float = 1.0,
                 omega0: float = 5.0) -> None:
        super(WaveletMFNNeRFSeparate, self).__init__()
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


class WaveletMFNNeRFPartial(nn.Module):
    """
    A NeRF model that integrates a partial positional encoding strategy.
    
    This model processes raw 3D points (keeping Euclidean distances meaningful for the
    wavelet filters) through separate WaveletMFN backbones for density and color. In the
    RGB head, the model reintroduces a positional encoding for the points (and also applies
    a positional encoding to the viewing direction) before predicting RGB colors.
    
    Args:
        hidden_features (int): Number of hidden features for both branches.
        density_hidden_layers (int): Number of hidden layers for the density branch.
        color_hidden_layers (int): Number of hidden layers for the color branch.
        point_encoding_dim (int): Number of frequencies for the positional encoding for points.
                                   (This encoding is only used in the RGB head.)
        dir_encoding_dim (int): Number of frequencies for the positional encoding of ray directions.
        sigma_mul (float): Multiplicative factor for the density output.
        rgb_mul (float): Multiplicative factor for the RGB output.
        omega0 (float): Frequency parameter used in both WaveletMFN networks.
    """
    def __init__(self, 
                 hidden_features: int = 256,
                 density_hidden_layers: int = 4,
                 color_hidden_layers: int = 4,
                 point_encoding_dim: int = 10,
                 dir_encoding_dim: int = 4,
                 sigma_mul: float = 1.0,
                 rgb_mul: float = 1.0,
                 omega0: float = 5.0) -> None:
        super(WaveletMFNNeRFPartial, self).__init__()
        self.point_encoding_dim = point_encoding_dim
        self.dir_encoding_dim = dir_encoding_dim
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul

        # Density branch using the WaveletMFN backbone on raw coordinates.
        self.density_mfn = WaveletMFN(
            in_features=3,
            hidden_features=hidden_features,
            out_features=1,
            hidden_layers=density_hidden_layers,
            omega0=omega0
        )
        
        # Color branch using the WaveletMFN backbone on raw coordinates.
        # The output dimensionality is set equal to hidden_features.
        self.color_mfn = WaveletMFN(
            in_features=3,
            hidden_features=hidden_features,
            out_features=hidden_features,
            hidden_layers=color_hidden_layers,
            omega0=omega0
        )
        
        # Compute the sizes for the positional encodings.
        # (For each dimension, the positional encoding expands the input to: 6 * encoding_dim + 3)
        point_pe_size = point_encoding_dim * 6 + 3
        dir_pe_size = dir_encoding_dim * 6 + 3
        
        # RGB head:
        # Combines the output of the color branch (raw features), the reintroduced positional encoding
        # of the points, and the encoded ray directions.
        rgb_input_dim = hidden_features + point_pe_size + dir_pe_size
        self.rgb_head = nn.Sequential(
            nn.Linear(rgb_input_dim, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 3)
        )

    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
            - Processes raw coordinates through density and color WaveletMFN backbones.
            - Applies a positional encoding to the ray directions and reintroduces a positional encoding
              for the points in the RGB head.
            - Predicts density (using a ReLU and scaling) and RGB color (using an RGB head followed
              by sigmoid activation and scaling).
        
        Args:
            points (torch.Tensor): Input 3D coordinates, shape (N, 3).
            rays_d (torch.Tensor): Ray directions, shape (N, 3).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - rgb: Predicted RGB colors, shape (N, 3).
                - density: Predicted densities, shape (N,).
        """
        # Density prediction:
        density_raw = self.density_mfn(points)  # (N, 1)
        density = torch.relu(density_raw) * self.sigma_mul
        
        # Color branch:
        features = self.color_mfn(points)  # (N, hidden_features)
        # Reintroduce a positional encoding for the points (partial encoding)
        points_enc = positional_encoding(points, self.point_encoding_dim)
        # Encode ray directions using positional encoding
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)
        
        # Concatenate the color branch features with both point and ray direction encodings.
        rgb_input = torch.cat((features, points_enc, rays_d_enc), dim=-1)
        rgb = self.rgb_head(rgb_input)
        # Scale and restrict the RGB values to the [0,1] range.
        rgb = torch.sigmoid(rgb * self.rgb_mul)
        
        return rgb, density.squeeze(-1)


class MultiScaleWaveletNeRF(nn.Module):
    def __init__(self, 
                 point_dim: int = 3,
                 dir_dim: int = 3,
                 num_freqs_dir: int = 4,
                 hidden_features: int = 512,
                 hidden_layers: int = 5,
                 alpha: float = 0.05,
                 beta: float = 0.025,
                 low_omega0: float = 0.1,
                 high_omega0: float = 5.0, # Was 10.0
                 sigma_mul: float = 1.0,
                 rgb_mul: float = 1.0):
        """
        A NeRF model using a multi-scale wavelet backbone.
        
        Args:
            point_dim: Dimensionality of spatial coordinates (usually 3).
            dir_dim: Dimensionality of ray directions (usually 3).
            num_freqs_dir: Number of frequency bands for ray direction encoding.
            hidden_features: Number of filters/features in each wavelet layer.
            hidden_layers: Number of wavelet layers in the backbone.
            alpha, beta: Gamma distribution parameters for wavelet filters.
            low_omega0, high_omega0: ω₀ values for low-frequency and high-frequency groups.
            sigma_mul, rgb_mul: Multiplicative factors for density and color outputs.
        """
        super(MultiScaleWaveletNeRF, self).__init__()
        self.num_freqs_dir = num_freqs_dir
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul
        
        # Build the backbone from multiple MultiScaleWaveletFilter layers.
        layers = []
        in_dim = point_dim  # initial input is the raw 3D point
        for i in range(hidden_layers):
            layers.append(MultiScaleWaveletFilter(in_dim, hidden_features, alpha, beta, low_omega0, high_omega0))
            in_dim = hidden_features  # update input dim for subsequent layers
        self.backbone = nn.Sequential(*layers)
        
        # Density head: maps backbone features to a density scalar.
        self.density_head = nn.Linear(hidden_features, 1)
        
        # Color head: combines backbone features with positional encodings of both points and ray directions.
        # We encode directions using the same positional encoding function.
        # For an input x, the encoding yields: x + [sin, cos, sin, cos, ...] → dimension: x_dim + 2*x_dim*num_freqs.
        dir_encoding_dim = dir_dim + 2 * dir_dim * num_freqs_dir
        
        color_input_dim = hidden_features + dir_encoding_dim
        
        self.color_head = nn.Sequential(
            nn.Linear(color_input_dim, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 3)
        )
        
    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            points: Tensor of shape (N, 3) containing 3D point coordinates.
            rays_d: Tensor of shape (N, 3) containing ray directions.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (RGB colors of shape (N, 3), density of shape (N,))
        """
        # Extract features using the backbone (multi-scale wavelet filters).
        features = self.backbone(points)
        # Density prediction: enforce non-negativity with ReLU and apply scaling.
        density = torch.relu(self.density_head(features)) * self.sigma_mul
        
        # Positional encodings for points and ray directions.
        rays_d_enc = positional_encoding(rays_d, self.num_freqs_dir)
        
        # Concatenate backbone features with both positional encodings.
        rgb_input = torch.cat([features, rays_d_enc], dim=-1)
        rgb = torch.sigmoid(self.color_head(rgb_input) * self.rgb_mul)
        
        return rgb, density.squeeze(-1)


class WaveletNeRFLearnable(nn.Module):
    """
    NeRF model using multi-scale wavelet filters.
    
    This model keeps the same overall architecture as the original WaveletNeRFLearnable:
      - A base MLP processes 3D points using stacked multi-scale wavelet layers.
      - A density branch predicts volume density.
      - A feature remapping branch and an RGB head combine backbone features with a positional encoding
        of the ray directions.
    
    Args:
        num_layers (int): Number of multi-scale wavelet layers in the base MLP.
        hidden_dim (int): Hidden dimension of the MLP.
        dir_encoding_dim (int): Number of frequency bands for ray direction encoding.
        sigma_mul (float): Multiplicative factor for density output.
        rgb_mul (float): Multiplicative factor for the RGB output.
        alpha (float): Scaling parameter for initializing gamma (per filter) in the wavelet filters.
        beta (float): Rate parameter for initializing gamma.
        low_omega0 (float): ω₀ value for the low-frequency group.
        high_omega0 (float): ω₀ value for the high-frequency group.
    """
    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 256,
        dir_encoding_dim: int = 4,
        sigma_mul: float = 1.0,
        rgb_mul: float = 1.0,
        alpha: float = 2.0,
        beta: float = 1.0,
        low_omega0: float = 0.1,
        high_omega0: float = 10.0
    ) -> None:
        super(WaveletNeRFLearnable, self).__init__()
        self.dir_encoding_dim = dir_encoding_dim
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul
        
        # Base MLP: Process 3D points with a stack of multi-scale wavelet layers.
        base_layers = [MultiScaleWaveletFilter(3, hidden_dim, alpha, beta, low_omega0, high_omega0)]
        for _ in range(num_layers - 1):
            base_layers.append(MultiScaleWaveletFilter(hidden_dim, hidden_dim, alpha, beta, low_omega0, high_omega0))
        self.block1 = nn.Sequential(*base_layers)
        
        # Density branch: Maps backbone features to a scalar density.
        self.density_branch = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        # Feature remapping: Prepares features for the RGB head.
        self.feature_remap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # RGB head: Combines remapped features with a positional encoding of ray directions.
        ray_encoding_size = 6 * self.dir_encoding_dim + 3
        self.rgb_head = nn.Sequential(
            MultiScaleWaveletFilter(hidden_dim + ray_encoding_size, hidden_dim // 2, alpha, beta, low_omega0, high_omega0),
            nn.Linear(hidden_dim // 2, 3)
        )
        
    def forward(self, points: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process 3D points through the base MLP.
        base = self.block1(points)
        
        # Compute density from backbone features.
        sigma = self.density_branch(base)
        density = torch.relu(sigma) * self.sigma_mul
        
        # Remap features.
        features = self.feature_remap(base)
        # Encode ray directions using positional encoding.
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)
        # Concatenate the remapped features with the encoded ray directions.
        rgb_input = torch.cat((features, rays_d_enc), dim=-1)
        # Compute RGB colors.
        rgb = self.rgb_head(rgb_input)
        rgb = torch.sigmoid_(rgb * self.rgb_mul)
        return rgb, density.squeeze(-1)
