import torch
from torch import Tensor


def compute_accumulated_transmittance(betas: Tensor) -> Tensor:
    """
    Computes cumulative transmittance along rays.

    Args:
        betas (Tensor): Complement of alpha values (1 - alpha) with shape (B, N)
                        where B = batch size (number of rays) and N = number of samples per ray.
    Returns:
        Tensor: Accumulated transmittance with shape (B, N)
    """
    accum_trans = torch.cumprod(betas, dim=1)  # shape: (B, N), cumulative product along samples
    init = torch.ones(accum_trans.shape[0], 1, device=accum_trans.device)  # shape: (B, 1)
    return torch.cat((init, accum_trans[:, :-1]), dim=1)  # shape: (B, N)


def stratified_sampling(near: float, far: float, num_bins: int, device: str = 'cpu') -> Tensor:
    """
    Performs stratified sampling in the range [near, far].

    Args:
        near (float): Near bound.
        far (float): Far bound.
        num_bins (int): Number of coarse samples per ray.
        device (str): Computation device.

    Returns:
        Tensor: A tensor of stratified samples with shape (num_coarse,)
                where num_coarse is the number of samples per ray.
    """
    bins = torch.linspace(near, far, num_bins + 1, device=device)  # shape: (num_coarse+1,)
    lower = bins[:-1]  # shape: (num_coarse,)
    upper = bins[1:]   # shape: (num_coarse,)
    random_offsets = torch.rand(num_bins, device=device)  # shape: (num_coarse,)
    return lower + (upper - lower) * random_offsets  # shape: (num_coarse,)


def normalize_directions(directions: Tensor) -> Tensor:
    """
    Normalizes ray directions to unit vectors.

    Args:
        directions (Tensor): Ray directions with shape (..., 3).

    Returns:
        Tensor: Normalized ray directions with the same shape as input.
    """
    return directions / torch.norm(directions, dim=-1, keepdim=True)  # shape: same as directions


def normalize_positions(positions: Tensor, near: float, far: float) -> Tensor:
    """
    Normalizes positions to the range [-1, 1].

    Args:
        positions (Tensor): Sampled positions with shape (..., 3).
        near (float): Near bound.
        far (float): Far bound.

    Returns:
        Tensor: Normalized positions with the same shape as input.
    """
    return 2 * (positions - near) / (far - near) - 1  # shape: same as positions


def generate_coarse_positions(
    rays_o_batch: Tensor,  # shape: (B, 3) where B = number of rays in the batch
    rays_d_batch: Tensor,  # shape: (B, 3) (normalized ray directions)
    near: float,
    far: float,
    num_coarse: int,
    device: str = 'cpu'
):
    """
    Generates stratified sample positions for a batch of rays and computes corresponding deltas.

    Returns:
        Tuple[Tensor, Tensor]:
            - sample_positions: Sample positions for each ray with shape (B, num_coarse, 3).
            - deltas: Deltas between adjacent sample positions with shape (num_coarse,)
                      (the last delta is set to a large value to indicate infinity).
    """
    # Get stratified samples along one ray
    stratified_samples = stratified_sampling(near, far, num_coarse, device)  # shape: (num_coarse,)
    # Compute deltas from stratified samples (difference between consecutive samples)
    deltas = stratified_samples[1:] - stratified_samples[:-1]  # shape: (num_coarse - 1,)
    delta_inf = torch.tensor([1e10], device=deltas.device, dtype=deltas.dtype)  # shape: (1,)
    deltas = torch.cat([deltas, delta_inf], dim=0)  # shape: (num_coarse,)
    # Compute sample positions for each ray in the batch:
    # Expand rays_o_batch: (B, 1, 3) and stratified_samples: (1, num_coarse, 1) so broadcasting works
    sample_positions = rays_o_batch.unsqueeze(1) + stratified_samples.unsqueeze(0).unsqueeze(-1) * rays_d_batch.unsqueeze(1)  # shape: (B, num_coarse, 3)
    return sample_positions, deltas  # sample_positions: (B, num_coarse, 3); deltas: (num_coarse,)


def query_model(model, sample_positions_flat: Tensor, directions_flat: Tensor, near: float, far: float):
    """
    Normalizes sample positions and queries the model to obtain colors and densities.

    Args:
        sample_positions_flat (Tensor): Flattened sample positions with shape (N, 3) where N = B * num_coarse.
        directions_flat (Tensor): Flattened ray directions corresponding to the sample positions with shape (N, 3).

    Returns:
        Tuple[Tensor, Tensor]:
            - colors_flat: Predicted colors with shape (N, 3)
            - densities_flat: Predicted densities with shape (N,)
    """
    # Normalize positions to the model's expected input range
    sample_positions_normalized = normalize_positions(sample_positions_flat, near, far)  # shape: (N, 3)
    return model.forward(sample_positions_normalized, directions_flat)  # shapes: (N, 3) and (N,)


def composite_volume(colors: Tensor, densities: Tensor, deltas: Tensor, white_background: bool) -> Tensor:
    """
    Performs volumetric rendering (alpha compositing) to compute final RGB colors for each ray.

    Args:
        colors (Tensor): Colors predicted by the model with shape (B, num_coarse, 3).
        densities (Tensor): Densities predicted by the model with shape (B, num_coarse).
        deltas (Tensor): Deltas between adjacent sample positions with shape (num_coarse,)
                      (or (num_coarse+1,) if different sampling is used).
        white_background (bool): If True, composite with a white background.

    Returns:
        Tensor: Composite RGB colors for each ray with shape (B, 3).
    """
    # Compute alpha values using the volume rendering formula
    # densities: (B, num_coarse), deltas.unsqueeze(0): (1, num_coarse)
    alpha = 1 - torch.exp(-densities * deltas.unsqueeze(0))  # shape: (B, num_coarse)
    # Compute weights along the ray:
    # (1 - alpha): (B, num_coarse) -> compute_accumulated_transmittance returns (B, num_coarse)
    weights = compute_accumulated_transmittance(1 - alpha) * alpha  # shape: (B, num_coarse)
    # Composite the colors by summing weighted contributions along the ray
    # weights.unsqueeze(-1): (B, num_coarse, 1); colors: (B, num_coarse, 3)
    comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)  # shape: (B, 3)
    if white_background:
        # weights.sum(dim=1, keepdim=True): (B, 1); added to each channel of comp_rgb
        comp_rgb = comp_rgb + (1 - weights.sum(dim=1, keepdim=True))  # shape: (B, 3)
    return comp_rgb  # shape: (B, 3)


def render_nerf(
    model,
    rays_o,           # shape: (N, 3) where N = total number of rays
    rays_d,           # shape: (N, 3)
    near,
    far,
    num_coarse=64,
    device='cpu',
    white_background=True,
    chunk_size=8192  # Process rays in chunks (e.g., 8192 rays per chunk)
):
    """
    Render rays via volumetric rendering using a NeRF model.

    Returns:
        Tensor: Rendered RGB colors for each ray with shape (N, 3).
    """
    rays_o = rays_o.to(device)  # shape: (N, 3)
    rays_d = normalize_directions(rays_d.to(device))  # shape: (N, 3)
    
    coarse_rgb = []  # List to hold rendered RGB values from each chunk, each element has shape: (chunk_size, 3)
    for i in range(0, rays_o.shape[0], chunk_size):
        # Process rays in chunks
        rays_o_chunk = rays_o[i:i + chunk_size]  # shape: (chunk_size, 3) (last chunk may be smaller)
        rays_d_chunk = rays_d[i:i + chunk_size]  # shape: (chunk_size, 3)

        # Generate stratified sample positions and deltas for the chunk
        coarse_positions, coarse_deltas = generate_coarse_positions(
            rays_o_chunk,   # shape: (chunk_size, 3)
            rays_d_chunk,   # shape: (chunk_size, 3)
            near, far, num_coarse, device
        )
        # coarse_positions shape: (chunk_size, num_coarse, 3)
        # coarse_deltas shape: (num_coarse,)
        coarse_positions_flat = coarse_positions.reshape(-1, 3)  # shape: (chunk_size * num_coarse, 3)
        coarse_directions_flat = rays_d_chunk.unsqueeze(1).expand(-1, num_coarse, -1).reshape(-1, 3)  # shape: (chunk_size * num_coarse, 3)

        # Query the model to obtain predicted colors and densities
        coarse_colors_flat, coarse_densities_flat = query_model(
            model,
            coarse_positions_flat,  # shape: (chunk_size * num_coarse, 3)
            coarse_directions_flat,  # shape: (chunk_size * num_coarse, 3)
            near, far
        )
        # coarse_colors_flat shape: (chunk_size * num_coarse, 3)
        # coarse_densities_flat shape: (chunk_size * num_coarse,)
        coarse_colors = coarse_colors_flat.reshape(rays_o_chunk.shape[0], num_coarse, 3)  # shape: (chunk_size, num_coarse, 3)
        coarse_densities = coarse_densities_flat.reshape(rays_o_chunk.shape[0], num_coarse)  # shape: (chunk_size, num_coarse)

        # Perform volumetric rendering via alpha compositing to obtain final RGB per ray
        coarse_comp_rgb = composite_volume(coarse_colors, coarse_densities, coarse_deltas, white_background)  # shape: (chunk_size, 3)
        coarse_rgb.append(coarse_comp_rgb)  # List element has shape: (chunk_size, 3)

    return torch.cat(coarse_rgb, dim=0)  # Final output shape: (N, 3)
