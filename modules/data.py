"""Shared scene loading and ray generation for Blender and LLFF datasets."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from modules.datasets import load_scene


def load_configured_scene(config, *, splits=('train', 'val', 'test')):
    return load_scene(
        config['dataset_path'], config.get('dataset_type', 'blender'),
        factor=int(config.get('dataset_factor', 1)) *
               (2 if config.get('half_res', 'false').lower() == 'true' else 1), splits=splits,
        white_background=str(config.get('white_background',
                                       'false' if config.get('dataset_type') == 'llff' else 'true')).lower() == 'true',
        llff_holdout=int(config.get('llff_holdout', 8)),
        llff_bounds_scale=float(config.get('llff_bounds_scale', 0.75)),
        llff_recenter=str(config.get('llff_recenter', 'true')).lower() == 'true',
        num_render_poses=int(config.get('num_render_poses', 80)),
        testskip=int(config.get('testskip', 1)),
    )


def resolve_sampling_bounds(config, scene):
    """Resolve 'auto' once and persist the concrete bounds in the run config."""
    near, far = (float(default if config.get(name, 'auto') == 'auto' else config[name])
                 for name, default in zip(('near', 'far'), scene.sampling_bounds))
    if scene.dataset_type == 'llff':
        if (near, far) != (0.0, 1.0):
            raise ValueError('Reference LLFF NDC requires near = 0 and far = 1')
    elif not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError('Sampling bounds must satisfy finite 0 < near < far')
    config.update(near=str(near), far=str(far))
    return near, far


def load_dataset(dataset_path: str, mode: str = 'train', single_image: bool = False,
                 *, white_background=True, half_res=False, testskip=1):
    """Legacy Blender tuple adapter; new callers should use load_scene instead."""
    split = load_scene(dataset_path, splits=(mode,), num_render_poses=1,
                       white_background=white_background, factor=2 if half_res else 1,
                       testskip=testskip).split(mode)
    selection = slice(0, 1) if single_image else slice(None)
    return split.images[selection], split.poses[selection], float(split.intrinsics[0, 0, 0])


def camera_rays(height, width, c2w_matrices, intrinsics, *, normalize=True):
    """Generate unit world rays using per-camera pinhole intrinsics (+Y up, -Z forward)."""
    poses = np.asarray(c2w_matrices, dtype=np.float32)
    count = len(poses)
    matrices = np.asarray(intrinsics, dtype=np.float32)
    if matrices.ndim == 0:
        focal = float(matrices)
        matrices = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                            dtype=np.float32)
    if matrices.shape == (3, 3):
        matrices = np.broadcast_to(matrices, (count, 3, 3))
    if poses.shape != (count, 4, 4) or matrices.shape != (count, 3, 3):
        raise ValueError('Rays require N x 4 x 4 poses and N x 3 x 3 intrinsics')
    if (count == 0 or height < 1 or width < 1 or not np.isfinite(matrices).all()
            or (matrices[:, (0, 1), (0, 1)] <= 0).any()):
        raise ValueError('Rays require nonempty images and finite positive focal lengths')
    u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32), indexing='xy')
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    camera_directions = np.einsum('nij,hwj->nhwi', np.linalg.inv(matrices), pixels)
    camera_directions[..., 1:] *= -1
    directions = np.einsum('nij,nhwj->nhwi', poses[:, :3, :3], camera_directions)
    if normalize:
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(poses[:, None, None, :3, 3], (count, height, width, 3))
    return origins.reshape(count, -1, 3).copy(), directions.reshape(count, -1, 3)


def compute_rays(images: np.ndarray, c2w_matrices: np.ndarray,
                 focal_length, *, normalize=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit world rays and RGB targets; accept scalar or matrix intrinsics."""
    count, height, width, _ = images.shape
    origins, directions = camera_rays(height, width, c2w_matrices, focal_length, normalize=normalize)
    return origins, directions, images.reshape(count, -1, 3)


class RayDataset(Dataset):
    """
    A PyTorch dataset for storing and accessing ray data.

    This dataset class converts ray origins, ray directions, and target pixel colors from 
    NumPy arrays to torch tensors. It flattens the input arrays to produce a dataset 
    where each sample corresponds to a single ray along with its associated color.

    Args:
        rays_o (np.ndarray): Array of ray origins with shape (N, H*W, 3).
        rays_d (np.ndarray): Array of ray directions with shape (N, H*W, 3).
        target_pixels (np.ndarray): Array of target RGB pixel colors with shape (N, H*W, 3).
    """
    def __init__(self, rays_o, rays_d, target_pixels, view_directions=None):
        self.rays_o = torch.from_numpy(rays_o.reshape(-1, 3)).float()
        self.rays_d = torch.from_numpy(rays_d.reshape(-1, 3)).float()
        self.target_pixels = torch.from_numpy(target_pixels.reshape(-1, 3)).float()
        self.view_directions = (torch.from_numpy(view_directions.reshape(-1, 3)).float()
                                if view_directions is not None else None)

    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, idx):
        ray = self.rays_o[idx], self.rays_d[idx], self.target_pixels[idx]
        return (*ray, self.view_directions[idx]) if self.view_directions is not None else ray
