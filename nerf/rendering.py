import torch
from torch import Tensor


def compute_accumulated_transmittance(betas: Tensor) -> Tensor:
    """
    Compute cumulative transmittance along rays.

    Args:
        betas (Tensor): Complement of alpha values (1 - alpha) for each sample.

    Returns:
        Tensor: Accumulated transmittance along each ray.
    """
    # Compute the cumulative product of (1 - alpha) values along the ray
    accum_trans = torch.cumprod(betas, dim=1)
    # Prepend a 1 to represent full transmittance at the ray origin
    init = torch.ones(accum_trans.shape[0], 1, device=accum_trans.device)
    return torch.cat((init, accum_trans[:, :-1]), dim=1)


def stratified_sampling(near: float, far: float, num_bins: int, device: str = 'cpu') -> Tensor:
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
    # Create bin edges across the depth range
    bins = torch.linspace(near, far, num_bins + 1, device=device)
    lower = bins[:-1]
    upper = bins[1:]
    # Randomly sample within each bin
    random_offsets = torch.rand(num_bins, device=device)
    return lower + (upper - lower) * random_offsets


def normalize_positions(positions: Tensor, near: float, far: float) -> Tensor:
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


def generate_sample_positions(
    rays_o_batch: Tensor,
    rays_d_batch: Tensor,
    near: float,
    far: float,
    num_samples: int,
    device: str = 'cpu'
):
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
        Tuple[Tensor, Tensor]: Sample positions for each ray and corresponding intervals.
    """
    # Obtain stratified samples along a single ray
    stratified_samples = stratified_sampling(near, far, num_samples, device)
    # Compute intervals between consecutive samples
    deltas = stratified_samples[1:] - stratified_samples[:-1]
    # Append a large value for the final interval to indicate infinity
    delta_inf = torch.tensor([1e10], device=deltas.device, dtype=deltas.dtype)
    deltas = torch.cat([deltas, delta_inf], dim=0)
    # Compute sample positions by advancing along each ray using the stratified samples
    sample_positions = rays_o_batch.unsqueeze(1) + stratified_samples.unsqueeze(0).unsqueeze(-1) * rays_d_batch.unsqueeze(1)
    return sample_positions, deltas


def query_model(model, sample_positions_flat: Tensor, directions_flat: Tensor, near: float, far: float):
    """
    Normalize sample positions and query the model to obtain colors and densities.

    Args:
        model: NeRF model.
        sample_positions_flat (Tensor): Flattened sample positions.
        directions_flat (Tensor): Flattened ray directions.
        near (float): Near bound for normalization.
        far (float): Far bound for normalization.

    Returns:
        Tuple[Tensor, Tensor]: Predicted colors and densities.
    """
    # Normalize positions to match the model's expected input range
    sample_positions_normalized = normalize_positions(sample_positions_flat, near, far)
    return model.forward(sample_positions_normalized, directions_flat)


def composite_volume(colors: Tensor, densities: Tensor, deltas: Tensor, white_background: bool) -> Tensor:
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
    # Compute alpha values from densities and intervals
    alpha = 1 - torch.exp(-densities * deltas.unsqueeze(0))
    # Calculate weights by accumulating transmittance along the ray
    weights = compute_accumulated_transmittance(1 - alpha) * alpha
    # Sum weighted colors along the ray to get final color
    comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)
    # Optionally composite with a white background for rays with low opacity
    if white_background:
        comp_rgb = comp_rgb + (1 - weights.sum(dim=1, keepdim=True))
    return comp_rgb


def render_nerf(
    model,
    rays_o,
    rays_d,
    near,
    far,
    num_samples=256,
    device='cpu',
    white_background=True,
    chunk_size=8192
):
    """
    Render rays via volumetric rendering using a NeRF model.

    Args:
        model: NeRF model.
        rays_o (Tensor): Ray origins.
        rays_d (Tensor): Ray directions.
        near (float): Near bound.
        far (float): Far bound.
        num_samples (int, optional): Number of samples per ray.
        device (str, optional): Computation device.
        white_background (bool, optional): Use a white background.
        chunk_size (int, optional): Number of rays to process per chunk.

    Returns:
        Tensor: Rendered RGB colors for each ray.
    """
    # Move rays to the designated device and normalize directions
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    coarse_rgb = []
    # Process rays in manageable chunks to conserve memory
    for i in range(0, rays_o.shape[0], chunk_size):
        # Select a chunk of rays
        rays_o_chunk = rays_o[i:i + chunk_size]
        rays_d_chunk = rays_d[i:i + chunk_size]

        # Generate stratified sample positions and compute sample intervals
        coarse_positions, coarse_deltas = generate_sample_positions(
            rays_o_chunk, rays_d_chunk, near, far, num_samples, device
        )
        # Flatten sample positions and duplicate directions to query the model
        coarse_positions_flat = coarse_positions.reshape(-1, 3)
        coarse_directions_flat = rays_d_chunk.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 3)

        # Query the model to obtain predicted colors and densities
        coarse_colors_flat, coarse_densities_flat = query_model(
            model, coarse_positions_flat, coarse_directions_flat, near, far
        )
        # Reshape model outputs back to the original chunk dimensions
        coarse_colors = coarse_colors_flat.reshape(rays_o_chunk.shape[0], num_samples, 3)
        coarse_densities = coarse_densities_flat.reshape(rays_o_chunk.shape[0], num_samples)

        # Composite the colors using the computed weights along the ray
        coarse_comp_rgb = composite_volume(coarse_colors, coarse_densities, coarse_deltas, white_background)
        coarse_rgb.append(coarse_comp_rgb)

    # Concatenate the results from all chunks and return the final rendered image
    return torch.cat(coarse_rgb, dim=0)
