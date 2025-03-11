import torch
from torch import Tensor
from tqdm import tqdm
from typing import Tuple


def stratified_sampling(
    near: float,
    far: float,
    num_bins: int,
    device: str = 'cpu'
) -> Tensor:
    """
    Perform stratified sampling within a given depth range.

    Args:
        near (float): Near bound of the sampling range.
        far (float): Far bound of the sampling range.
        num_bins (int): Number of samples.
        device (str): Device for computation.

    Returns:
        Tensor: Stratified samples within [near, far].
    """
    bins = torch.linspace(near, far, num_bins + 1, device=device)
    lower = bins[:-1]
    upper = bins[1:]
    random_offsets = torch.rand(num_bins, device=device)
    return lower + (upper - lower) * random_offsets


def generate_sample_positions(
    rays_o_batch: Tensor,
    rays_d_batch: Tensor,
    near: float,
    far: float,
    num_samples: int,
    device: str = 'cpu'
) -> Tuple[Tensor, Tensor]:
    """
    Generate stratified sample positions for a batch of rays and compute sample intervals.

    Args:
        rays_o_batch (Tensor): Ray origins.
        rays_d_batch (Tensor): Ray directions.
        near (float): Near bound for sampling.
        far (float): Far bound for sampling.
        num_samples (int): Number of samples per ray.
        device (str): Computation device.

    Returns:
        Tuple[Tensor, Tensor]:
            - sample_positions (Tensor): Sample positions for each ray.
            - deltas (Tensor): Intervals between sampled positions.
    """
    strat_samples = stratified_sampling(near, far, num_samples, device)
    deltas = strat_samples[1:] - strat_samples[:-1]

    # Append a large interval for the last sample
    delta_inf = torch.tensor([1e10], device=deltas.device, dtype=deltas.dtype)
    deltas = torch.cat([deltas, delta_inf], dim=0)

    # Expand each ray to get sample positions
    sample_positions = (
        rays_o_batch.unsqueeze(1)
        + strat_samples.unsqueeze(0).unsqueeze(-1) * rays_d_batch.unsqueeze(1)
    )

    return sample_positions, deltas


def normalize_positions(
    positions: Tensor,
    near: float,
    far: float
) -> Tensor:
    """
    Normalize positions to the range [-1, 1].

    Args:
        positions (Tensor): Sampled positions.
        near (float): Near bound.
        far (float): Far bound.

    Returns:
        Tensor: Normalized positions.
    """
    return 2 * (positions - near) / (far - near) - 1


def query_model(
    model: torch.nn.Module,
    sample_positions_flat: Tensor,
    directions_flat: Tensor,
    near: float,
    far: float
) -> Tuple[Tensor, Tensor]:
    """
    Normalize sample positions and query the model to obtain colors and densities.

    Args:
        model (torch.nn.Module): NeRF model.
        sample_positions_flat (Tensor): Flattened sample positions.
        directions_flat (Tensor): Flattened ray directions.
        near (float): Near bound for normalization.
        far (float): Far bound for normalization.

    Returns:
        Tuple[Tensor, Tensor]: 
            - colors (Tensor): Predicted colors.
            - densities (Tensor): Predicted densities.
    """
    sample_positions_normalized = normalize_positions(sample_positions_flat, near, far)
    return model.forward(sample_positions_normalized, directions_flat)


def compute_accumulated_transmittance(betas: Tensor) -> Tensor:
    """
    Compute cumulative transmittance along rays.

    Args:
        betas (Tensor): Complement of alpha values (1 - alpha) for each sample.

    Returns:
        Tensor: Accumulated transmittance along each ray.
    """
    accum_trans = torch.cumprod(betas, dim=1)
    # Prepend a 1 so the transmittance for the first sample is 1
    init = torch.ones(accum_trans.shape[0], 1, device=accum_trans.device)
    return torch.cat((init, accum_trans[:, :-1]), dim=1)


def composite_volume(
    colors: Tensor,
    densities: Tensor,
    deltas: Tensor,
    white_background: bool
) -> Tensor:
    """
    Composite colors along each ray using volumetric rendering.

    Args:
        colors (Tensor): Colors predicted by the model.
        densities (Tensor): Densities predicted by the model.
        deltas (Tensor): Intervals between sampled positions.
        white_background (bool): Flag indicating whether to composite with a white background.

    Returns:
        Tensor: Composite RGB colors for each ray.
    """
    # alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1 - torch.exp(-densities * deltas.unsqueeze(0))

    # weights_i = T_i * alpha_i
    weights = compute_accumulated_transmittance(1 - alpha) * alpha

    # Final composite color
    comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)

    # If we assume a white background, add the remainder
    if white_background:
        comp_rgb = comp_rgb + (1 - weights.sum(dim=1, keepdim=True))

    return comp_rgb


def render_nerf(
    model: torch.nn.Module,
    rays_o: Tensor,
    rays_d: Tensor,
    near: float,
    far: float,
    num_samples: int = 256,
    device: str = 'cpu',
    white_background: bool = True,
    chunk_size: int = 8192,
    show_progress: bool = False,
) -> Tensor:
    """
    Render rays via volumetric rendering using a NeRF model.
    
    ... [docstring truncated for brevity]
    """
    # Ensure data is on the specified device
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)

    rgb_out = []
    
    # Choose the iterator: wrap with tqdm if progress is enabled
    if show_progress:
        iterator = tqdm(range(0, rays_o.shape[0], chunk_size),
                        desc="Rendering",
                        position=1,
                        leave=True)
    else:
        iterator = range(0, rays_o.shape[0], chunk_size)
        
    for i in iterator:
        rays_o_chunk = rays_o[i:i + chunk_size]
        rays_d_chunk = rays_d[i:i + chunk_size]

        # Sample positions along each ray
        sample_positions, deltas = generate_sample_positions(
            rays_o_chunk,
            rays_d_chunk,
            near,
            far,
            num_samples,
            device
        )

        # Flatten for batching into the network
        sample_positions_flat = sample_positions.reshape(-1, 3)
        directions_flat = (
            rays_d_chunk
            .unsqueeze(1)
            .expand(-1, num_samples, -1)
            .reshape(-1, 3)
        )

        # Query the model for colors & densities
        colors_flat, densities_flat = query_model(
            model,
            sample_positions_flat,
            directions_flat,
            near,
            far
        )
        
        # Reshape back to [batch, num_samples, ...]
        colors = colors_flat.reshape(rays_o_chunk.shape[0], num_samples, 3)
        densities = densities_flat.reshape(rays_o_chunk.shape[0], num_samples)

        # Composite final color per ray
        composed_rgb = composite_volume(colors, densities, deltas, white_background)
        rgb_out.append(composed_rgb)

    # Concatenate results for all chunks
    return torch.cat(rgb_out, dim=0)
