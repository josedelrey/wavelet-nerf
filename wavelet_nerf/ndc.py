"""Reference NeRF NDC projection for forward-facing LLFF geometry rays.

Adapted from bmild/nerf/run_nerf_helpers.py (MIT; notice preserved in LICENSE).
The projection near plane is distinct from the subsequent sampling bounds [0, 1].
"""

import numpy as np

from wavelet_nerf.data import camera_rays


def project_rays_ndc(rays_o, rays_d, height, width, focal, near=1.0, *, focal_y=None,
                     cx=None, cy=None):
    """Project world rays to NDC without normalizing the output directions."""
    focal_y = focal if focal_y is None else focal_y
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    if not np.isfinite([height, width, near]).all() or min(height, width, near) <= 0:
        raise ValueError('NDC projection requires positive finite image dimensions and near plane')
    if not np.isfinite(focal).all() or not np.isfinite(focal_y).all() \
            or np.any(np.asarray(focal) <= 0) or np.any(np.asarray(focal_y) <= 0):
        raise ValueError('NDC focal lengths must be finite and positive')
    if not np.isfinite(rays_o).all() or not np.isfinite(rays_d).all():
        raise ValueError('Cannot project nonfinite rays')
    if not np.isfinite(cx).all() or not np.isfinite(cy).all():
        raise ValueError('NDC principal points must be finite')
    if np.any(np.abs(rays_d[..., 2]) < 1e-8):
        raise ValueError('Cannot project rays parallel to the NDC near plane')
    distance = -(near + rays_o[..., 2]) / rays_d[..., 2]
    origin = rays_o + distance[..., None] * rays_d
    x_scale, y_scale = -2 * focal / width, -2 * focal_y / height
    o0 = x_scale * origin[..., 0] / origin[..., 2] + 2 * cx / width - 1
    o1 = y_scale * origin[..., 1] / origin[..., 2] + 1 - 2 * cy / height
    o2 = 1 + 2 * near / origin[..., 2]
    d0 = x_scale * (rays_d[..., 0] / rays_d[..., 2] - origin[..., 0] / origin[..., 2])
    d1 = y_scale * (rays_d[..., 1] / rays_d[..., 2] - origin[..., 1] / origin[..., 2])
    d2 = -2 * near / origin[..., 2]
    return np.stack((o0, o1, o2), axis=-1), np.stack((d0, d1, d2), axis=-1)


def ndc_camera_rays(height, width, poses, intrinsics):
    """Return projected geometry rays and original unit world appearance directions."""
    origins, viewdirs = camera_rays(height, width, poses, intrinsics)
    matrices = np.asarray(intrinsics)
    if matrices.ndim == 2:
        fx, fy = matrices[0, 0], matrices[1, 1]
        cx, cy = matrices[0, 2], matrices[1, 2]
    else:
        fx, fy = matrices[:, 0, 0, None], matrices[:, 1, 1, None]
        cx, cy = matrices[:, 0, 2, None], matrices[:, 1, 2, None]
    origins, directions = project_rays_ndc(origins, viewdirs, height, width, fx, focal_y=fy, cx=cx, cy=cy)
    return origins, directions, viewdirs
