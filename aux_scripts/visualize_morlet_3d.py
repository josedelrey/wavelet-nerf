import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go

# ——— Your exact 3D Morlet filter class ———
class MorletWaveletFilter3D(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, alpha=1.0, beta=1.0, omega0=5.0):
        super().__init__()
        # learnable centers in [-1,1]^3
        self.mu    = nn.Parameter(torch.rand(out_dim, in_dim)*2 - 1)
        # learnable gamma sampled from Gamma(α,β)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim,))
        )
        # linear projection before cosine
        self.linear = nn.Linear(in_dim, out_dim)
        # learnable frequency
        self.omega0 = nn.Parameter(torch.tensor(omega0))
        self._init_weights()

    def _init_weights(self):
        # exactly as in your init: scale weights & uniform-phase bias
        self.linear.weight.data  *= 128.0 * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def morlet(self, u):
        # ψ(u) = cos(ω₀ u) – exp(–ω₀²/2)
        return torch.cos(self.omega0 * u) - torch.exp(-0.5 * (self.omega0**2))

    def forward(self, x):
        # squared distance ‖x–μ‖²
        d2 = (x.pow(2).sum(1,keepdim=True)
             + self.mu.pow(2).sum(1).unsqueeze(0)
             - 2*x @ self.mu.t())
        # Gaussian envelope
        env = torch.exp(-0.5 * self.gamma.unsqueeze(0) * d2)
        # cosine carrier
        car = self.morlet(self.linear(x))
        return (env * car).squeeze(-1)   # shape (N,)

# ——— Parameters you can tweak ———
alpha   = 6.0      # Gamma shape
beta    = 1.0      # Gamma rate
omega0  = 5.0      # morlet frequency
res     = 80       # resolution per axis
iso_value = 0.2    # isosurface cutoff

# instantiate single-filter layer
filter3d = MorletWaveletFilter3D(3, 1, alpha, beta, omega0)

# sample a uniform grid in [-1,1]^3
axis = np.linspace(-1, 1, res)
X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
pts_t = torch.tensor(pts, dtype=torch.float32)

# compute the full Morlet response over the grid
with torch.no_grad():
    resp = filter3d(pts_t).numpy().reshape(res, res, res)

# build Plotly isosurface
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=resp.flatten(),
    isomin=iso_value,        # lower contour
    isomax=resp.max(),       # upper
    surface_count=1,
    colorscale='Viridis',
    caps=dict(x_show=False, y_show=False, z_show=False),
))
fig.update_layout(
    title=f'3D Morlet Wavelet Isosurface (α={alpha},β={beta},ω₀={omega0})',
    scene=dict(aspectmode='cube')
)
fig.show()
