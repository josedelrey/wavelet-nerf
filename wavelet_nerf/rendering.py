import torch
import numpy as np
from torch import Tensor
from typing import Tuple

from wavelet_nerf.scene import SceneNormalization
from wavelet_nerf.models import NeRF, LegacyNeRF
from wavelet_nerf.nerf_reference import render_reference_nerf, sample_depths
from wavelet_nerf.render_validation import validate_bounds, validate_count, validate_rays, validate_sampling_options


def stratified_sampling(
    near: float,
    far: float,
    num_bins: int,
    device: str = 'cpu', *, num_rays: int | None = None,
    dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Perform stratified sampling within a given depth range.

    Args:
        near (float): Near bound of the sampling range.
        far (float): Far bound of the sampling range.
        num_bins (int): Number of samples.
        device (str): Device for computation.

    Returns:
        Tensor: Stratified samples within [near, far].
            Shape is [num_rays, num_bins] when num_rays is supplied; each ray
            receives independent offsets. Otherwise returns one depth vector.
    """
    validate_bounds(near, far)
    validate_count(num_bins, 'num_bins')
    if num_rays is not None:
        validate_count(num_rays, 'num_rays')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Depth sampling requires float32 or float64 geometry')
    bins = torch.linspace(near, far, num_bins + 1, device=device, dtype=dtype)
    lower = bins[:-1]
    upper = bins[1:]
    shape = (num_bins,) if num_rays is None else (num_rays, num_bins)
    random_offsets = torch.rand(shape, device=device, dtype=dtype)
    return lower + (upper - lower) * random_offsets


def generate_sample_positions(
    rays_o_batch: Tensor,
    rays_d_batch: Tensor,
    near: float,
    far: float,
    num_samples: int,
    device: str = 'cpu', reference: bool = True) -> Tuple[Tensor, Tensor]:
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
            - deltas (Tensor): Intervals in ray-parameter space; multiply by the
              geometry direction norm before volume integration.
    """
    validate_rays(rays_o_batch, rays_d_batch)
    validate_bounds(near, far)
    validate_count(num_samples, 'num_samples', minimum=2 if reference else 1)
    depths = (sample_depths(rays_d_batch, near, far, num_samples, perturb=True)
              if reference else stratified_sampling(near, far, num_samples, rays_d_batch.device,
                                                    num_rays=len(rays_d_batch), dtype=rays_d_batch.dtype))
    deltas = torch.cat((depths[:, 1:] - depths[:, :-1],
                        torch.full_like(depths[:, :1], 1e10)), dim=-1)
    sample_positions = rays_o_batch[:, None] + depths[..., None] * rays_d_batch[:, None]

    return sample_positions, deltas


def normalize_positions(
    positions: Tensor,
    scene_normalization: SceneNormalization) -> Tensor:
    """
    Map the configured scene cube to [-1, 1] without clipping outside points.

    Args:
        positions (Tensor): Sampled positions.
        scene_normalization (SceneNormalization): World-space center and half-extent.

    Returns:
        Tensor: Normalized positions.
    """
    center = positions.new_tensor(scene_normalization.center)
    return (positions - center) / scene_normalization.scale


def query_model(
    model: torch.nn.Module,
    sample_positions_flat: Tensor,
    directions_flat: Tensor,
    scene_normalization: SceneNormalization) -> Tuple[Tensor, Tensor]:
    """
    Normalize sample positions and query the model to obtain colors and densities.

    Args:
        model (torch.nn.Module): NeRF model.
        sample_positions_flat (Tensor): Flattened sample positions.
        directions_flat (Tensor): Flattened ray directions.
        scene_normalization (SceneNormalization): Scene coordinates used by the model.

    Returns:
        Tuple[Tensor, Tensor]: 
            - colors (Tensor): Predicted colors.
            - densities (Tensor): Predicted densities.
    """
    sample_positions_normalized = normalize_positions(sample_positions_flat, scene_normalization)
    return model(sample_positions_normalized, directions_flat)


def compute_accumulated_transmittance(betas: Tensor) -> Tensor:
    """
    Compute accumulated transmittance along rays.

    Args:
        betas (Tensor): Complement of alpha values (1 - alpha) for each sample.

    Returns:
        Tensor: Accumulated transmittance along each ray.
    """
    accum_trans = torch.cumprod(betas, dim=1)
    init = torch.ones_like(accum_trans[:, :1])
    return torch.cat((init, accum_trans[:, :-1]), dim=1)


def composite_volume(
    colors: Tensor,
    densities: Tensor,
    deltas: Tensor,
    white_background: bool) -> Tensor:
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
    # Keep integration at least float32 even if a network query is autocast.
    dtype = torch.promote_types(torch.promote_types(colors.dtype, densities.dtype), deltas.dtype)
    dtype = torch.promote_types(dtype, torch.float32)
    colors, densities, deltas = (value.to(dtype) for value in (colors, densities, deltas))
    # alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1 - torch.exp(-densities * deltas)

    # weights_i = T_i * alpha_i
    weights = compute_accumulated_transmittance(1 - alpha) * alpha

    comp_rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)

    if white_background:
        comp_rgb = comp_rgb + (1 - weights.sum(dim=1, keepdim=True))

    return comp_rgb


def render_nerf(
    model: torch.nn.Module,
    rays_o: Tensor,
    rays_d: Tensor,
    near: float,
    far: float,
    num_samples: int | None = None,
    device: str = 'cpu',
    white_background: bool = True,
    chunk_size: int = 8192,
    stratified: bool = True,
    scene_normalization: SceneNormalization = SceneNormalization(),
    num_importance=None, netchunk=65536, perturb=1.0, lindisp=False,
    raw_noise_std=0.0, return_aux=False,
    view_directions: Tensor | None = None,
    output_device=None) -> Tensor | dict[str, Tensor]:
    """
    Render rays with a NeRF model via volumetric integration.

    This function samples points along each ray between the specified near and far
    bounds, queries the NeRF model to obtain color and density predictions, and
    composites these predictions into a final RGB color per ray using volumetric rendering.

    Parameters:
        model (torch.nn.Module): The NeRF model that outputs colors and densities.
        rays_o (Tensor): Ray origins of shape [num_rays, 3].
        rays_d (Tensor): Ray directions of shape [num_rays, 3].
        near (float): Near bound for sampling along the rays.
        far (float): Far bound for sampling along the rays.
        num_samples (int, optional): Coarse sample count; defaults to 64 for
            reference NeRF and 256 for other models.
        device (str, optional): Device on which to perform rendering.
        white_background (bool, optional): If True, composite over a white background.
        chunk_size (int, optional): Number of rays processed per chunk for memory efficiency.
        stratified (bool, optional): If True, use stratified sampling (default), 
                                     else use uniform sampling for validation.
        scene_normalization (SceneNormalization): World-space scene transform,
                                                  independent of near/far; defaults to identity.
        view_directions (Tensor, optional): Original unit world directions for
            appearance when geometry rays are in NDC. Defaults to unit rays_d.
        Geometry rays must be finite, nonzero float32/float64 N x 3 tensors.
            Directions need not have unit length: integration multiplies depth
            intervals by their norm. Bounds parameterize origins + depth * directions.
        num_importance (int, optional): Additional fine samples for reference NeRF.
        return_aux (bool): Return reference coarse/fine outputs for the two losses.
            Scene normalization is ignored for reference NeRF, matching upstream.
        output_device: Optional inference output destination (e.g. 'cpu').
            Completed chunks are copied into a preallocated output buffer.
            Requires disabled gradients. Otherwise chunks retain their graphs
            until concatenation/backpropagation; chunk_size does not cap that memory.

    Returns:
        Tensor: A tensor of shape [num_rays, 3] containing the rendered RGB colors.
    """
    validate_count(netchunk, 'netchunk')
    validate_count(chunk_size, 'chunk_size')
    validate_bounds(near, far, lindisp=lindisp)
    validate_rays(rays_o, rays_d, view_directions)
    validate_sampling_options(perturb, raw_noise_std)
    if output_device is not None and torch.is_grad_enabled():
        raise ValueError('output_device is for inference; use torch.no_grad()')
    if view_directions is None:
        view_directions = torch.nn.functional.normalize(rays_d, dim=-1)
    else:
        if view_directions.shape != rays_d.shape:
            raise ValueError('Viewing directions must match geometry ray directions')

    ordinary_model = getattr(model, '_orig_mod', model)
    if num_samples is None:
        num_samples = 64 if isinstance(ordinary_model, NeRF) else 256
    validate_count(num_samples, 'num_samples', minimum=2)
    if num_importance is not None:
        validate_count(num_importance, 'num_importance', minimum=0)
    if isinstance(ordinary_model, NeRF):
        outputs = render_reference_nerf(
            model, rays_o, rays_d, near, far, num_samples=num_samples,
            num_importance=ordinary_model.num_importance if num_importance is None else num_importance,
            chunk_size=chunk_size, netchunk=netchunk, stratified=stratified,
            perturb=perturb, lindisp=lindisp, white_background=white_background,
            raw_noise_std=raw_noise_std, view_directions=view_directions,
            device=device, output_device=output_device,
        )
        return outputs if return_aux else outputs['rgb_map']

    rgb_out = []
    output_buffer = None
    for i in range(0, rays_o.shape[0], chunk_size):
        rays_o_chunk = rays_o[i:i + chunk_size].to(device)
        rays_d_chunk = rays_d[i:i + chunk_size].to(device)

        if stratified:
            # Use the existing stratified sampling to generate sample positions and deltas.
            sample_positions, deltas = generate_sample_positions(
                rays_o_chunk,
                rays_d_chunk,
                near,
                far,
                num_samples,
                device, reference=not isinstance(ordinary_model, LegacyNeRF)
            )
        else:
            # Uniform sampling: generate evenly spaced sample positions between near and far.
            samples = torch.linspace(near, far, num_samples, device=device, dtype=rays_d_chunk.dtype)
            
            # Compute intervals (deltas) between consecutive sample positions.
            deltas = samples[1:] - samples[:-1]
            delta_inf = torch.tensor([1e10], device=device, dtype=deltas.dtype)
            deltas = torch.cat([deltas, delta_inf], dim=0)

            # Compute the actual positions along the rays.
            sample_positions = (
                rays_o_chunk.unsqueeze(1)
                + samples.unsqueeze(0).unsqueeze(-1) * rays_d_chunk.unsqueeze(1)
            )

        sample_positions_flat = sample_positions.reshape(-1, 3)
        directions_flat = (
            view_directions[i:i + chunk_size]
            .to(device)
            .unsqueeze(1)
            .expand(-1, num_samples, -1)
            .reshape(-1, 3)
        )

        # Normalize positions and query the model to get colors and densities.
        queries = [query_model(model, sample_positions_flat[start:start + netchunk],
                               directions_flat[start:start + netchunk], scene_normalization)
                   for start in range(0, len(sample_positions_flat), netchunk)]
        colors_flat = torch.cat([colors for colors, _ in queries])
        densities_flat = torch.cat([densities for _, densities in queries])
        
        colors = colors_flat.reshape(rays_o_chunk.shape[0], num_samples, 3)
        densities = densities_flat.reshape(rays_o_chunk.shape[0], num_samples)

        # Composite the colors using the computed weights from the densities.
        # NDC directions remain unnormalized. As in the reference, convert
        # parameter intervals to distances in the geometry ray's coordinate space.
        distances = deltas * torch.linalg.vector_norm(rays_d_chunk, dim=-1, keepdim=True)
        composed_rgb = composite_volume(colors, densities, distances, white_background)
        if output_device is None:
            rgb_out.append(composed_rgb)
        else:
            if output_buffer is None:
                output_buffer = torch.empty((len(rays_o), 3), dtype=composed_rgb.dtype, device=output_device)
            output_buffer[i:i + len(composed_rgb)].copy_(composed_rgb.to(output_device))

    return torch.cat(rgb_out, dim=0) if output_device is None else output_buffer


@torch.no_grad()
def render_camera(model, height, width, pose, intrinsics, near, far, *, dataset_type='blender',
                  chunk_size=8192, device='cpu', **options):
    """Render one camera with chunk-sized ray/GPU storage and a CPU RGB buffer."""
    from wavelet_nerf.data import CameraRayGenerator
    ordinary = getattr(model, '_orig_mod', model)
    generator = CameraRayGenerator(height, width, np.asarray(pose).reshape(1, 4, 4),
                                   np.asarray(intrinsics).reshape(1, 3, 3), dataset_type=dataset_type,
                                   normalize=not isinstance(ordinary, NeRF))
    validate_count(chunk_size, 'chunk_size')
    output = None
    for start in range(0, height * width, chunk_size):
        pixels = np.arange(start, min(start + chunk_size, height * width))
        rays = generator.rays(np.zeros(len(pixels), dtype=np.int64), pixels)
        viewing = {'view_directions': torch.from_numpy(rays[2])} if len(rays) == 3 else {}
        rgb = render_nerf(model, torch.from_numpy(rays[0]), torch.from_numpy(rays[1]), near, far,
                          device=device, chunk_size=chunk_size,
                          **{**options, 'stratified': False, 'output_device': 'cpu'}, **viewing)
        if isinstance(rgb, dict):
            rgb = rgb['rgb_map']
        if output is None:
            output = torch.empty((height * width, 3), dtype=rgb.dtype)
        output[start:start + len(pixels)].copy_(rgb)
    return output
