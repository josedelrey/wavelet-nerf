import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define a simplified 2D Morlet wavelet filter
class WaveletFilter2D(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, alpha=1.0, beta=0.2, omega0=5.0):
        super().__init__()
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta)
            .sample((out_dim,))
        )
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega0 = nn.Parameter(torch.tensor(omega0))
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data *= 128.0 * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def morlet_wavelet(self, u):
        return torch.cos(self.omega0 * u) - torch.exp(-0.5 * (self.omega0**2))

    def forward(self, x):
        # Euclidean distance squared
        norm = (x**2).sum(dim=1).unsqueeze(-1) \
             + (self.mu**2).sum(dim=1).unsqueeze(0) \
             - 2 * x @ self.mu.T
        envelope = torch.exp(-self.gamma.unsqueeze(0) / 2 * norm)
        lin_out = self.linear(x)
        carrier = self.morlet_wavelet(lin_out)
        full = envelope * carrier
        return full, envelope, carrier

# Instantiate a 2-filter 2D wavelet layer
wavelet2d = WaveletFilter2D(in_dim=2, out_dim=2, alpha=6, beta=1, omega0=5.0)

# Sample a grid in the [-1,1] x [-1,1] plane
grid = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(grid, grid)
points = np.stack([X.ravel(), Y.ravel()], axis=1)
pts_t = torch.tensor(points, dtype=torch.float32)

# Compute responses
full_resp, envelope, carrier = wavelet2d(pts_t)

# Choose filter index to visualize
idx = 0

# Reshape to grid
Z_env = envelope[:, idx].detach().numpy().reshape(200, 200)
Z_full = full_resp[:, idx].detach().numpy().reshape(200, 200)

# Plot the Gaussian envelope
plt.figure()
plt.imshow(Z_env, extent=[-1,1,-1,1], origin='lower')
plt.colorbar()
plt.title('Gaussian Envelope (Filter 0)')
plt.xlabel('x')
plt.ylabel('y')

# Plot the full Morlet wavelet response
plt.figure()
plt.imshow(Z_full, extent=[-1,1,-1,1], origin='lower')
plt.colorbar()
plt.title('Full Morlet Wavelet Response (Filter 0)')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
