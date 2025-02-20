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
    Performs stratified sampling in the range [near, far] with num_bins.

    Args:
        near (float): Near bound for sampling.
        far (float): Far bound for sampling.
        num_bins (int): Number of samples per ray.
        device (str): Device on which to run computations.

    Returns:
        Tensor: Sampled positions.
    """
    bins = torch.linspace(near, far, num_bins + 1, device=device)
    lower = bins[:-1]
    upper = bins[1:]

    random_offsets = torch.rand(num_bins, device=device)  # Random values in [0, 1)
    stratified_samples = lower + (upper - lower) * random_offsets
    return stratified_samples


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
        positions (Tensor): Sampled positions in world space.
        near (float): Near bound for sampling.
        far (float): Far bound for sampling.

    Returns:
        Tensor: Normalized positions in the range [-1, 1].
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
    chunk_size=1024 * 8
):
    """
    Render a batch of rays using volumetric rendering with a NeRF model.
    
    This function implements the core volumetric rendering procedure:
    1. Stratified sampling along each ray within the bounds [near, far].
    2. Querying the NeRF model to obtain colors and densities for each sampled point.
    3. Applying the volumetric rendering (alpha compositing) equation to combine
       the sample predictions into a final pixel color.
    
    The process is computed in chunks to avoid excessive memory usage when dealing 
    with a large number of rays.
    
    Args:
        model: A neural network model (e.g., a NeRF) that predicts colors and densities.
        rays_o (Tensor): Ray origins with shape (N, 3).
        rays_d (Tensor): Ray directions with shape (N, 3).
        near (float): Near bound for sampling along the ray.
        far (float): Far bound for sampling along the ray.
        num_bins (int, optional): Number of stratified samples per ray. Default is 100.
        device (str, optional): The device for computation (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
        white_background (bool, optional): If True, composites the colors on a white background.
        chunk_size (int, optional): Maximum number of rays processed at one time to limit memory usage.
    
    Returns:
        Tensor: The rendered RGB colors for each ray, with shape (N, 3).
    """
    # Transfer ray origins and directions to the specified device.
    rays_o = rays_o.to(device)
    rays_d = normalize_directions(rays_d.to(device))
    
    rendered_rgb = []  # List to collect rendered colors from each chunk.

    # Process the rays in chunks to manage memory efficiently.
    for i in range(0, rays_o.shape[0], chunk_size):
        # Extract a chunk of ray origins and directions.
        rays_o_chunk = rays_o[i:i + chunk_size]
        rays_d_chunk = rays_d[i:i + chunk_size]

        # Perform stratified sampling along the rays in the range [near, far].
        stratified_samples = stratified_sampling(near, far, num_bins, device)
        
        # Compute the differences (deltas) between consecutive samples for integration.
        deltas = stratified_samples[1:] - stratified_samples[:-1]
        # Append a large delta for the last sample to simulate an infinite interval.
        delta_inf = torch.tensor([1e10], device=deltas.device, dtype=deltas.dtype)
        deltas = torch.cat([deltas, delta_inf], dim=0)

        # Compute 3D sample positions along each ray:
        # Each sample position = ray origin + (sample depth * ray direction)
        # Resulting shape: (chunk_size, num_bins, 3)
        sample_positions = (
            rays_o_chunk.unsqueeze(1) +  # Shape: (chunk_size, 1, 3)
            stratified_samples.unsqueeze(0).unsqueeze(-1) * rays_d_chunk.unsqueeze(1)
        )

        # Flatten the sample positions to shape (chunk_size * num_bins, 3) for model evaluation.
        sample_positions_flat = sample_positions.reshape(-1, 3)
        # Expand the ray directions so that each sample position has an associated direction.
        expanded_dirs = rays_d_chunk.unsqueeze(1).expand(-1, num_bins, -1).reshape(-1, 3)

        # Normalize sample positions to the range [-1, 1] if required by the model.
        sample_positions_flat = normalize_positions(sample_positions_flat, near, far)
        
        # Query the model to obtain predicted colors and densities at each sample position.
        colors_flat, densities_flat = model.forward(sample_positions_flat, expanded_dirs)

        # Reshape the flat outputs back to (chunk_size, num_bins, channels) format.
        colors = colors_flat.reshape(rays_o_chunk.shape[0], num_bins, 3)
        densities = densities_flat.reshape(rays_o_chunk.shape[0], num_bins)

        # Compute alpha values from densities using the formula:
        #   alpha = 1 - exp(-density * delta)
        alpha = 1 - torch.exp(-densities * deltas.unsqueeze(0))
        
        # Compute weights for each sample along the ray.
        # The weight is defined as the product of the sample's alpha and the accumulated transmittance:
        #   weight = T * alpha, where T = cumprod(1 - alpha) (with proper initialization).
        weights = compute_accumulated_transmittance(1 - alpha) * alpha

        # Composite the final color for each ray using the computed weights.
        if white_background:
            # For white background, add the background contribution based on the transparency.
            comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)
            weight_sum = weights.sum(dim=1, keepdim=True)
            comp_rgb = comp_rgb + (1 - weight_sum)
        else:
            # Otherwise, simply sum the weighted colors.
            comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)

        # Collect the rendered colors from this chunk.
        rendered_rgb.append(comp_rgb)

    # Concatenate the results from all chunks to form the final rendered image.
    return torch.cat(rendered_rgb, dim=0)
