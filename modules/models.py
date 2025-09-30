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
            nn.Linear(hidden_dim, hidden_dim + 1)
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
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self,
                points: torch.Tensor,
                rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        points_enc = positional_encoding(points, self.pos_encoding_dim)
        rays_d_enc = positional_encoding(rays_d, self.dir_encoding_dim)

        features = self.block1(points_enc)
        features = self.block2(torch.cat((features, points_enc), dim=1))

        density = self.softplus(features[:, -1])
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


class MFNBase(nn.Module):
    """
    MFNBase: Multiplicative filter network base class.
    
    This is the original implementation from "Multiplicative Filter Networks"
    by Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter (2021).
    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """
    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out
    

class WaveletLayer(nn.Module):
    """
    WaveletLayer: recomputes heavy distance matrix on each call.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_scale: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        omega0: float = 5.0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.omega0 = omega0

        # Scale linear weights by sqrt(gamma)
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        D = (
            x.pow(2).sum(-1, keepdim=True)
            + self.mu.pow(2).sum(-1).unsqueeze(0)
            - 2 * x @ self.mu.T
        )
        gabor = torch.sin(self.linear(self.omega0 * x))
        return gabor * torch.exp(-0.5 * D * self.gamma[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._core(x)


class WaveletNet(MFNBase):
    """
    WaveletNet: Network using WaveletLayer filters, with optional normalization.
    
    Args:
        in_features (int): Input feature dimension.
        hidden_features (int): Hidden feature dimension.
        out_features (int): Output feature dimension.
        hidden_layers (int): Number of multiplicative layers (default: 3).
        input_scale (float): Scale applied to input of WaveletLayer (default: 256.0).
        weight_scale (float): Scaling factor for linear weights (default: 1.0).
        alpha (float): Alpha parameter for WaveletLayer (default: 6.0).
        beta (float): Beta parameter for WaveletLayer (default: 1.0).
        omega0 (float): Omega0 parameter for WaveletLayer (default: 5.0).
        bias (bool): Whether to include bias in linear layers (default: True).
        output_act (bool): Whether to apply final sine activation (default: False).
        normalized (bool): Whether to apply LayerNorm on filter and linear outputs (default: False).
    """
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        omega0=5.0,
        normalized=False,
    ):
        # Initialize base linear and multiplicative branches
        super().__init__(hidden_features, out_features, hidden_layers, weight_scale)

        self.normalized = normalized
        n_layers = hidden_layers + 1
        scale = input_scale / np.sqrt(n_layers)
        alpha_scaled = alpha / n_layers

        # Create Wavelet filter layers
        self.filters = nn.ModuleList([
            WaveletLayer(
                in_features,
                hidden_features,
                scale,
                alpha_scaled,
                beta,
                omega0,
            )
            for _ in range(n_layers)
        ])

        if self.normalized:
            # LayerNorm for each filter and each linear branch
            self.filter_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_features) for _ in range(n_layers)]
            )
            self.linear_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_features) for _ in range(hidden_layers)]
            )

    def forward(self, x):
        # First filter pass
        out = self.filters[0](x)
        if self.normalized:
            out = self.filter_norms[0](out)

        # Subsequent multiplicative filter + linear branches
        for i in range(1, len(self.filters)):
            f = self.filters[i](x)
            if self.normalized:
                f = self.filter_norms[i](f)

            l = self.linear[i-1](out)
            if self.normalized:
                l = self.linear_norms[i-1](l)

            out = f * l

        # Final linear output
        out = self.output_linear(out)

        return out


class WaveletNeRF(nn.Module):
    """
    NeRF-style model using a WaveletNet base and simple linear RGB head.
    """
    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        dir_encoding_dim: int = 4,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
        alpha: float = 6.0,
        beta: float = 0.5,
        omega0: float = 5.0,
        normalized: bool = True
    ) -> None:
        super().__init__()
        self.dir_encoding_dim = dir_encoding_dim

        # Base MLP: WaveletNet processes 3D points
        self.base_net = WaveletNet(
            in_features=in_features,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            hidden_layers=num_layers,
            input_scale=input_scale,
            weight_scale=weight_scale,
            alpha=alpha,
            beta=beta,
            omega0=omega0,
            normalized=normalized,
        )

        # Density branch
        self.density_branch = nn.Linear(hidden_dim, 1)

        # Feature remap for RGB head
        self.feature_remap = nn.Linear(hidden_dim, hidden_dim)

        # Ray direction encoding size (positional_encoding should be defined elsewhere)
        ray_enc_size = dir_encoding_dim * 6 + in_features
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + ray_enc_size, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(
        self,
        points: torch.Tensor,
        rays_d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Base features
        base_feats = self.base_net(points)

        # Density
        raw_sigma = self.density_branch(base_feats)
        density = torch.relu(raw_sigma.squeeze(-1))

        # Rgb
        remapped = self.feature_remap(base_feats)
        dir_enc = positional_encoding(rays_d, self.dir_encoding_dim)
        rgb_in = torch.cat([remapped, dir_enc], dim=-1)
        rgb = self.rgb_head(rgb_in)
        return rgb, density
