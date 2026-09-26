"""PyTorch translation of the rendering equations in bmild/nerf.

Reference: run_nerf.py (render_rays) and run_nerf_helpers.py (sample_pdf).
The upstream MIT notice is preserved in LICENSE.
"""

import math

import torch
from wavelet_nerf.render_validation import validate_bounds, validate_count, validate_rays, validate_sampling_options


def sample_pdf(bins, weights, num_samples, deterministic=False):
    """Invert the piecewise-constant PDF, including upstream endpoint conventions."""
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[..., :1]), torch.cumsum(pdf, dim=-1)], dim=-1)
    shape = (*cdf.shape[:-1], num_samples)
    if deterministic:
        u = torch.linspace(0, 1, num_samples, device=bins.device, dtype=bins.dtype).expand(shape)
    else:
        u = torch.rand(shape, device=bins.device, dtype=bins.dtype)
    indices = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = (indices - 1).clamp_min(0)
    above = indices.clamp_max(cdf.shape[-1] - 1)
    cdf_lower, cdf_upper = torch.gather(cdf, -1, below), torch.gather(cdf, -1, above)
    bin_lower, bin_upper = torch.gather(bins, -1, below), torch.gather(bins, -1, above)
    denominator = cdf_upper - cdf_lower
    denominator = torch.where(denominator < 1e-5, torch.ones_like(denominator), denominator)
    return bin_lower + (u - cdf_lower) / denominator * (bin_upper - bin_lower)


def sample_depths(rays, near, far, num_samples, perturb=True, lindisp=False):
    """Endpoint-inclusive depths with independent midpoint-bin jitter per ray."""
    validate_bounds(near, far, lindisp=lindisp)
    validate_count(num_samples, 'num_samples', minimum=2)
    t = torch.linspace(0, 1, num_samples, device=rays.device, dtype=rays.dtype)
    depths = (1 / ((1 - t) / near + t / far) if lindisp
              else near * (1 - t) + far * t)
    depths = depths.expand(len(rays), -1)
    if perturb:
        mids = (depths[..., 1:] + depths[..., :-1]) * 0.5
        lower = torch.cat([depths[..., :1], mids], dim=-1)
        upper = torch.cat([mids, depths[..., -1:]], dim=-1)
        depths = lower + (upper - lower) * torch.rand_like(depths)
    return depths


def raw_to_outputs(raw, depths, rays_d, white_background=False, raw_noise_std=0.0):
    """Sigmoid RGB, noisy ReLU density, ray-length-corrected alpha compositing."""
    # Promote autocast network outputs before the large terminal interval and exp/cumprod.
    dtype = torch.promote_types(torch.promote_types(raw.dtype, depths.dtype),
                               torch.promote_types(rays_d.dtype, torch.float32))
    raw, depths, rays_d = (value.to(dtype) for value in (raw, depths, rays_d))
    distances = torch.cat([
        depths[..., 1:] - depths[..., :-1], torch.full_like(depths[..., :1], 1e10)
    ], dim=-1) * torch.linalg.vector_norm(rays_d, dim=-1, keepdim=True)
    density = raw[..., 3]
    if raw_noise_std > 0:
        density = density + torch.randn_like(density) * raw_noise_std
    alpha = 1 - torch.exp(-torch.relu(density) * distances)
    transmittance = torch.cumprod(torch.cat([
        torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-10
    ], dim=-1), dim=-1)[..., :-1]
    weights = alpha * transmittance
    rgb = (weights[..., None] * torch.sigmoid(raw[..., :3])).sum(dim=-2)
    depth = (weights * depths).sum(dim=-1)
    acc = weights.sum(dim=-1)
    # Match upstream, including its undefined disparity for wholly empty rays.
    disparity = 1 / torch.maximum(torch.full_like(depth, 1e-10), depth / acc)
    if white_background:
        rgb = rgb + (1 - acc[..., None])
    return {'rgb_map': rgb, 'disp_map': disparity, 'acc_map': acc,
            'depth_map': depth, 'weights': weights}


def query_network(model, positions, viewdirs, fine, netchunk):
    points = positions.reshape(-1, 3)
    directions = viewdirs[:, None, :].expand_as(positions).reshape(-1, 3)
    output = [model(points[i:i + netchunk], directions[i:i + netchunk],
                    fine=fine, return_raw=True)
              for i in range(0, len(points), netchunk)]
    return torch.cat(output).reshape(*positions.shape[:-1], 4)


def render_reference_nerf(model, rays_o, rays_d, near, far, num_samples=64,
                          num_importance=128, chunk_size=1024, netchunk=65536,
                          stratified=True, perturb=1.0, lindisp=False,
                          white_background=True, raw_noise_std=0.0,
                          view_directions=None, device=None, output_device=None):
    """Coarse/fine rendering with separate geometry and world viewing directions."""
    validate_count(num_samples, 'num_samples', minimum=2)
    validate_count(num_importance, 'num_importance', minimum=0)
    validate_count(chunk_size, 'chunk_size')
    validate_count(netchunk, 'netchunk')
    validate_bounds(near, far, lindisp=lindisp)
    validate_rays(rays_o, rays_d, view_directions)
    validate_sampling_options(perturb, raw_noise_std)
    if num_importance > 0 and num_samples < 3:
        raise ValueError('Reference rendering needs >=2 coarse samples (>=3 with importance sampling)')
    if output_device is not None and torch.is_grad_enabled():
        raise ValueError('output_device is for inference; use torch.no_grad()')
    device = rays_o.device if device is None else device
    results = {}
    jitter = stratified and perturb > 0
    noise = raw_noise_std if stratified else 0.0
    for start in range(0, len(rays_o), chunk_size):
        origins = rays_o[start:start + chunk_size].to(device)
        directions = rays_d[start:start + chunk_size].to(device)
        viewdirs = (directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
                    if view_directions is None else view_directions[start:start + chunk_size].to(device))
        depths = sample_depths(directions, near, far, num_samples, jitter, lindisp)
        positions = origins[:, None, :] + directions[:, None, :] * depths[..., None]
        coarse = raw_to_outputs(query_network(model, positions, viewdirs, False, netchunk),
                                depths, directions, white_background, noise)
        outputs = {key: coarse[key] for key in ('rgb_map', 'disp_map', 'acc_map', 'depth_map')}
        if num_importance > 0:
            mids = 0.5 * (depths[..., 1:] + depths[..., :-1])
            with torch.no_grad():
                importance = sample_pdf(mids, coarse['weights'][..., 1:-1],
                                        num_importance, deterministic=not jitter)
            merged = torch.sort(torch.cat([depths, importance], dim=-1), dim=-1).values
            positions = origins[:, None, :] + directions[:, None, :] * merged[..., None]
            fine = raw_to_outputs(query_network(model, positions, viewdirs, True, netchunk),
                                  merged, directions, white_background, noise)
            outputs = {key: fine[key] for key in ('rgb_map', 'disp_map', 'acc_map', 'depth_map')}
            outputs.update(rgb0=coarse['rgb_map'], disp0=coarse['disp_map'],
                           acc0=coarse['acc_map'], z_std=importance.std(dim=-1, unbiased=False))
        for key, value in outputs.items():
            if output_device is None:
                results.setdefault(key, []).append(value)
            else:
                if key not in results:
                    results[key] = torch.empty((len(rays_o), *value.shape[1:]),
                                               dtype=value.dtype, device=output_device)
                results[key][start:start + len(value)].copy_(value.to(output_device))
    return {key: torch.cat(value) for key, value in results.items()} if output_device is None else results


class KerasAdam(torch.optim.Optimizer):
    """Dense Adam with TF 1.15 Keras's epsilon-hat placement (epsilon=1e-7).

    Implemented from the documented update equations, rather than changing the
    PyTorch Adam epsilon, which has different bias-correction semantics.
    """

    def __init__(self, params, lr=5e-4, betas=(0.9, 0.999), eps=1e-7):
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(eps) or eps < 0:
            raise ValueError('Adam learning rate and epsilon must be finite and nonnegative')
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError('Adam betas must lie in [0, 1)')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError('KerasAdam only supports dense gradients')
                state = self.state[parameter]
                if not state:
                    state.update(step=torch.tensor(0.), exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state['step'] += 1
                step = float(state['step'])
                state['exp_avg'].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                rate = group['lr'] * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                parameter.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(group['eps']), value=-rate)
        return loss
