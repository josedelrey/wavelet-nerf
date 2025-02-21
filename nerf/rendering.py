import torch
from torch import Tensor


def compute_accumulated_transmittance(betas: Tensor) -> Tensor:
    """
    Computes cumulative transmittance along rays.

    Args:
        betas (Tensor): Complement of alpha values (1 - alpha).

    Returns:
        Tensor: Accumulated transmittance.
    """
    accum_trans = torch.cumprod(betas, dim=1)
    init = torch.ones(accum_trans.shape[0], 1, device=accum_trans.device)
    return torch.cat((init, accum_trans[:, :-1]), dim=1)


def stratified_sampling(near: float, far: float, num_bins: int, device: str = 'cpu') -> Tensor:
    """
    Performs stratified sampling in the range [near, far].

    Args:
        near (float): Near bound.
        far (float): Far bound.
        num_bins (int): Number of samples per ray.
        device (str): Computation device.

    Returns:
        Tensor: Sampled positions.
    """
    bins = torch.linspace(near, far, num_bins + 1, device=device)
    lower = bins[:-1]
    upper = bins[1:]
    random_offsets = torch.rand(num_bins, device=device)
    return lower + (upper - lower) * random_offsets


def normalize_directions(directions: Tensor) -> Tensor:
    """
    Normalizes ray directions to unit vectors.

    Args:
        directions (Tensor): Ray directions.

    Returns:
        Tensor: Normalized ray directions.
    """
    return directions / torch.norm(directions, dim=-1, keepdim=True)


def normalize_positions(positions: Tensor, near: float, far: float) -> Tensor:
    """
    Normalizes positions to the range [-1, 1].

    Args:
        positions (Tensor): Sampled positions.
        near (float): Near bound.
        far (float): Far bound.

    Returns:
        Tensor: Normalized positions.
    """
    return 2 * (positions - near) / (far - near) - 1


def render_volume(
    model,
    rays_o,
    rays_d,
    near,
    far,
    num_bins=100,
    device='cpu',
    white_background=True,
    chunk_size=8192  # 1024 * 8
):
    """
    Render rays via volumetric rendering using a NeRF model.

    This function performs stratified sampling, queries the model for
    color and density predictions, and applies alpha compositing to
    produce final RGB values.

    Args:
        model: Neural network predicting colors and densities.
        rays_o (Tensor): Ray origins (N, 3).
        rays_d (Tensor): Ray directions (N, 3).
        near (float): Near bound for sampling.
        far (float): Far bound for sampling.
        num_bins (int, optional): Number of samples per ray. Default is 100.
        device (str, optional): Computation device. Default is 'cpu'.
        white_background (bool, optional): Use white background for compositing.
        chunk_size (int, optional): Number of rays processed per chunk.

    Returns:
        Tensor: Rendered RGB colors for each ray (N, 3).
    """
    rays_o = rays_o.to(device)
    rays_d = normalize_directions(rays_d.to(device))
    
    rendered_rgb = []
    for i in range(0, rays_o.shape[0], chunk_size):
        rays_o_chunk = rays_o[i:i + chunk_size]
        rays_d_chunk = rays_d[i:i + chunk_size]

        stratified_samples = stratified_sampling(near, far, num_bins, device)
        deltas = stratified_samples[1:] - stratified_samples[:-1]
        delta_inf = torch.tensor([1e10], device=deltas.device, dtype=deltas.dtype)
        deltas = torch.cat([deltas, delta_inf], dim=0)

        sample_positions = rays_o_chunk.unsqueeze(1) + stratified_samples.unsqueeze(0).unsqueeze(-1) * rays_d_chunk.unsqueeze(1)
        sample_positions_flat = sample_positions.reshape(-1, 3)
        expanded_dirs = rays_d_chunk.unsqueeze(1).expand(-1, num_bins, -1).reshape(-1, 3)

        # Normalize positions to the model's expected input range
        sample_positions_flat = normalize_positions(sample_positions_flat, near, far)
        
        colors_flat, densities_flat = model.forward(sample_positions_flat, expanded_dirs)
        colors = colors_flat.reshape(rays_o_chunk.shape[0], num_bins, 3)
        densities = densities_flat.reshape(rays_o_chunk.shape[0], num_bins)

        alpha = 1 - torch.exp(-densities * deltas.unsqueeze(0))
        weights = compute_accumulated_transmittance(1 - alpha) * alpha

        # Perform alpha compositing
        if white_background:
            comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)
            comp_rgb = comp_rgb + (1 - weights.sum(dim=1, keepdim=True))
        else:
            comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)

        rendered_rgb.append(comp_rgb)

    return torch.cat(rendered_rgb, dim=0)
