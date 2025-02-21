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
        Tensor: A tensor of stratified samples.
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


def generate_sample_positions(
    rays_o_batch: Tensor,
    rays_d_batch: Tensor,
    near: float,
    far: float,
    num_bins: int,
    device: str = 'cpu'
):
    """
    Generates stratified sample positions for a batch of rays and computes corresponding deltas.

    Args:
        rays_o_batch (Tensor): Batch of ray origins (B, 3).
        rays_d_batch (Tensor): Batch of normalized ray directions (B, 3).
        near (float): Near bound.
        far (float): Far bound.
        num_bins (int): Number of samples per ray.
        device (str): Computation device.

    Returns:
        Tuple[Tensor, Tensor]:
            - sample_positions: Sample positions for each ray (B, num_bins, 3).
            - deltas: Deltas between adjacent sample positions (num_bins+1,).
    """
    # Get stratified samples along one ray
    stratified_samples = stratified_sampling(near, far, num_bins, device)
    # Compute deltas from stratified samples
    deltas = stratified_samples[1:] - stratified_samples[:-1]
    delta_inf = torch.tensor([1e10], device=deltas.device, dtype=deltas.dtype)
    deltas = torch.cat([deltas, delta_inf], dim=0)
    # Compute sample positions for each ray in the batch
    sample_positions = rays_o_batch.unsqueeze(1) + stratified_samples.unsqueeze(0).unsqueeze(-1) * rays_d_batch.unsqueeze(1)
    return sample_positions, deltas


def query_model(model, sample_positions_flat: Tensor, directions_flat: Tensor, near: float, far: float):
    """
    Normalizes sample positions and queries the model to obtain colors and densities.

    Args:
        model: Neural network predicting colors and densities.
        sample_positions_flat (Tensor): Flattened sample positions (N, 3).
        directions_flat (Tensor): Flattened ray directions corresponding to the sample positions (N, 3).
        near (float): Near bound.
        far (float): Far bound.

    Returns:
        Tuple[Tensor, Tensor]: colors_flat (N, 3) and densities_flat (N,)
    """
    # Normalize positions to the model's expected input range
    sample_positions_normalized = normalize_positions(sample_positions_flat, near, far)
    return model.forward(sample_positions_normalized, directions_flat)


def composite_volume(colors: Tensor, densities: Tensor, deltas: Tensor, white_background: bool) -> Tensor:
    """
    Performs volumetric rendering (alpha compositing) to compute final RGB colors for each ray.

    Args:
        colors (Tensor): Colors predicted by the model (B, num_bins, 3).
        densities (Tensor): Densities predicted by the model (B, num_bins).
        deltas (Tensor): Deltas between adjacent sample positions (num_bins+1,).
        white_background (bool): If True, composite with a white background.

    Returns:
        Tensor: Composite RGB colors for each ray (B, 3).
    """
    # Compute alpha values
    alpha = 1 - torch.exp(-densities * deltas.unsqueeze(0))
    # Compute weights along the ray
    weights = compute_accumulated_transmittance(1 - alpha) * alpha
    # Composite the colors
    comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)
    if white_background:
        comp_rgb = comp_rgb + (1 - weights.sum(dim=1, keepdim=True))
    return comp_rgb


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

    This function performs the following steps:
      1. Normalizes ray directions.
      2. Splits the rays into chunks.
      3. For each chunk:
         - Generates stratified sample positions and deltas.
         - Flattens sample positions and expands ray directions.
         - Queries the model to get colors and densities.
         - Applies volumetric rendering (alpha compositing) to compute final RGB values.

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
        # Process rays in chunks
        rays_o_chunk = rays_o[i:i + chunk_size]
        rays_d_chunk = rays_d[i:i + chunk_size]

        # Generate stratified sample positions and compute deltas
        sample_positions, deltas = generate_sample_positions(rays_o_chunk, rays_d_chunk, near, far, num_bins, device)
        # Flatten sample positions and expand directions for the model query
        sample_positions_flat = sample_positions.reshape(-1, 3)
        expanded_dirs = rays_d_chunk.unsqueeze(1).expand(-1, num_bins, -1).reshape(-1, 3)

        # Query the model (MLP call) to obtain colors and densities
        colors_flat, densities_flat = query_model(model, sample_positions_flat, expanded_dirs, near, far)
        colors = colors_flat.reshape(rays_o_chunk.shape[0], num_bins, 3)
        densities = densities_flat.reshape(rays_o_chunk.shape[0], num_bins)

        # Perform volumetric rendering via alpha compositing
        comp_rgb = composite_volume(colors, densities, deltas, white_background)
        rendered_rgb.append(comp_rgb)

    return torch.cat(rendered_rgb, dim=0)
