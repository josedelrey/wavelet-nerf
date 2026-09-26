"""Shared scene loading and ray generation for Blender and LLFF datasets."""

from typing import Tuple
from dataclasses import replace
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from wavelet_nerf.datasets import load_scene
from wavelet_nerf.datasets.images import validate_rgb_images
from wavelet_nerf.datasets.types import validate_cameras
from wavelet_nerf.camera import configured_render_path, render_camera_path


def load_configured_scene(config, *, splits=('train', 'val', 'test')):
    scene = load_scene(
        config['dataset_path'], config.get('dataset_type', 'blender'),
        factor=int(config.get('dataset_factor', 1)) *
               (2 if config.get('half_res', False) else 1), splits=splits,
        white_background=config.get('white_background', config.get('dataset_type') != 'llff'),
        llff_holdout=int(config.get('llff_holdout', 8)),
        llff_bounds_scale=float(config.get('llff_bounds_scale', 0.75)),
        llff_recenter=config.get('llff_recenter', True),
        num_render_poses=int(config.get('num_render_poses', 80)),
        testskip=int(config.get('testskip', 1)),
    )
    path = configured_render_path(scene.render_path, config)
    return replace(scene, render_path=path,
                   render_poses=render_camera_path(path, int(config.get('num_render_poses', 80))))


def resolve_sampling_bounds(config, scene):
    """Resolve 'auto' once and persist the concrete bounds in the run config."""
    near, far = (float(default if config.get(name, 'auto') == 'auto' else config[name])
                 for name, default in zip(('near', 'far'), scene.sampling_bounds))
    if scene.dataset_type == 'llff':
        if (near, far) != (0.0, 1.0):
            raise ValueError('Reference LLFF NDC requires near = 0 and far = 1')
    elif not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError('Sampling bounds must satisfy finite 0 < near < far')
    config.update(near=near, far=far)
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
    if poses.ndim != 3:
        raise ValueError('Camera poses must be a nonempty N x 4 x 4 array')
    count = len(poses)
    matrices = np.asarray(intrinsics, dtype=np.float32)
    if matrices.ndim == 0:
        focal = float(matrices)
        matrices = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                            dtype=np.float32)
    if matrices.shape == (3, 3):
        matrices = np.broadcast_to(matrices, (count, 3, 3))
    _validate_dimensions(height, width)
    validate_cameras(poses, matrices)
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
    validate_rgb_images(images)
    count, height, width, _ = images.shape
    origins, directions = camera_rays(height, width, c2w_matrices, focal_length, normalize=normalize)
    if len(origins) != count:
        raise ValueError(f'Image/pose count mismatch: {count} images, {len(origins)} poses')
    return origins, directions, images.reshape(count, -1, 3)


def _validate_dimensions(height, width):
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           or value < 1 for value in (height, width)):
        raise ValueError('Image height and width must be positive integers')


class CameraRayGenerator:
    """Generate selected pixel rays, caching only per-camera matrices."""

    def __init__(self, height, width, poses, intrinsics, *, dataset_type='blender', normalize=True):
        self.height, self.width = height, width
        self.poses = np.asarray(poses, dtype=np.float32)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        _validate_dimensions(height, width)
        if self.poses.ndim != 3:
            raise ValueError('Camera poses must be a nonempty N x 4 x 4 array')
        if self.intrinsics.shape == (3, 3):
            self.intrinsics = np.broadcast_to(self.intrinsics, (len(self.poses), 3, 3))
        validate_cameras(self.poses, self.intrinsics)
        if dataset_type not in ('blender', 'llff'):
            raise ValueError('Pixel rays require blender or llff cameras')
        self.inverse_intrinsics = np.linalg.inv(self.intrinsics)
        self.dataset_type, self.normalize = dataset_type, normalize

    def rays(self, camera_indices, pixel_indices):
        camera_indices = np.asarray(camera_indices, dtype=np.int64)
        pixel_indices = np.asarray(pixel_indices, dtype=np.int64)
        if camera_indices.shape != pixel_indices.shape or camera_indices.ndim != 1:
            raise ValueError('Camera and pixel indices must be matching one-dimensional arrays')
        if (camera_indices < 0).any() or (camera_indices >= len(self.poses)).any() \
                or (pixel_indices < 0).any() or (pixel_indices >= self.height * self.width).any():
            raise ValueError('Camera or pixel index is outside the scene')
        pixels = np.stack((pixel_indices % self.width, pixel_indices // self.width,
                           np.ones(len(pixel_indices))), axis=-1).astype(np.float32)
        directions = np.einsum('bij,bj->bi', self.inverse_intrinsics[camera_indices], pixels)
        directions[:, 1:] *= -1
        directions = np.einsum('bij,bj->bi', self.poses[camera_indices, :3, :3], directions)
        if self.normalize or self.dataset_type == 'llff':
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        origins = self.poses[camera_indices, :3, 3].copy()
        if self.dataset_type == 'llff':
            # Local import avoids the existing ndc -> data camera-helper dependency.
            from wavelet_nerf.ndc import project_rays_ndc
            matrices = self.intrinsics[camera_indices]
            ndc_origins, ndc_directions = project_rays_ndc(
                origins, directions, self.height, self.width, matrices[:, 0, 0],
                focal_y=matrices[:, 1, 1], cx=matrices[:, 0, 2], cy=matrices[:, 1, 2])
            return ndc_origins, ndc_directions, directions
        return origins, directions


class PixelRaySampler:
    """Keep images and camera matrices; create geometry only for a sampled batch.

    Pixels are sampled without replacement within each batch. Batches are
    independent, rather than consuming a full-scene shuffled ray permutation.
    """

    def __init__(self, images, poses, intrinsics, *, image_indices=None,
                 dataset_type='blender', normalize=True, seed=42):
        validate_rgb_images(images)
        self.images = images
        self.height, self.width = images.shape[1:3]
        self.camera_indices = (np.arange(len(images)) if image_indices is None
                               else np.asarray(image_indices, dtype=np.int64))
        if len(self.camera_indices) == 0:
            raise ValueError('Training must have at least one image')
        self.generator = CameraRayGenerator(self.height, self.width, poses, intrinsics,
                                            dataset_type=dataset_type, normalize=normalize)
        if len(self.generator.poses) != len(images):
            raise ValueError('Image/pose count mismatch in pixel sampler')
        if self.camera_indices.ndim != 1 or (self.camera_indices < 0).any() \
                or (self.camera_indices >= len(images)).any():
            raise ValueError('Training image indices must select existing views')
        self.rng = np.random.default_rng(seed)
        self.commit()

    def commit(self, rng_state=None):
        """Advance checkpoint progress only after a successful optimizer update."""
        self._committed_rng = deepcopy(self.rng.bit_generator.state if rng_state is None else rng_state)

    def state_dict(self):
        return {'protocol': 'independent_pixel_batches_v1', 'height': self.height,
                'width': self.width, 'camera_indices': self.camera_indices.tolist(),
                'rng': deepcopy(self._committed_rng)}

    def load_state_dict(self, state):
        if state.get('protocol') != 'independent_pixel_batches_v1' \
                or state['height'] != self.height or state['width'] != self.width \
                or state['camera_indices'] != self.camera_indices.tolist():
            raise ValueError('Checkpoint pixel sampler protocol or scene differs from this run')
        self.rng.bit_generator.state = deepcopy(state['rng'])
        self.commit()

    def plan(self, size, *, per_image=False, crop_fraction=None):
        if size < 1:
            raise ValueError('Training batch size must be positive')
        if per_image:
            camera = self.rng.choice(self.camera_indices)
            if crop_fraction is None:
                top, left, height, width = 0, 0, self.height, self.width
            else:
                if not 0 < crop_fraction <= 1:
                    raise ValueError('Crop fraction must be between 0 and 1')
                dh, dw = int(self.height // 2 * crop_fraction), int(self.width // 2 * crop_fraction)
                top, left, height, width = self.height // 2 - dh, self.width // 2 - dw, 2 * dh, 2 * dw
            if size > height * width:
                raise ValueError('num_random_rays exceeds the available image/crop pixels')
            selected = self.rng.choice(height * width, size=size, replace=False, shuffle=False)
            pixels = (selected // width + top) * self.width + selected % width + left
            cameras = np.full(size, camera, dtype=np.int64)
        else:
            total = len(self.camera_indices) * self.height * self.width
            selected = self.rng.choice(total, size=min(size, total), replace=False, shuffle=False)
            cameras = self.camera_indices[selected // (self.height * self.width)]
            pixels = selected % (self.height * self.width)
        return {'cameras': cameras, 'pixels': pixels, 'rng': deepcopy(self.rng.bit_generator.state)}

    def batch(self, cameras, pixels):
        rays = self.generator.rays(cameras, pixels)
        targets = self.images[cameras, pixels // self.width, pixels % self.width]
        return (*rays[:2], targets, *rays[2:])

    def sample(self, size, *, per_image=False, crop_fraction=None):
        plan = self.plan(size, per_image=per_image, crop_fraction=crop_fraction)
        return self.batch(plan['cameras'], plan['pixels'])


class RayBatchDataset(Dataset):
    """Workers construct geometry for a small parent-selected pixel plan."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __getitem__(self, plan):
        batch = self.sampler.batch(plan['cameras'], plan['pixels'])
        return {'rays': tuple(torch.from_numpy(array) for array in batch), 'rng': plan['rng']}


class PixelBatchPlans(Sampler):
    """Prefetch indices only; saved progress follows the consumed plan's RNG."""

    def __init__(self, sampler, size, start, end, *, per_image=False, precrop_iters=0, precrop_frac=.5):
        self.sampler, self.size, self.start, self.end = sampler, size, start, end
        self.per_image, self.precrop_iters, self.precrop_frac = per_image, precrop_iters, precrop_frac

    def __iter__(self):
        for step in range(self.start, self.end):
            crop = self.precrop_frac if self.per_image and step < self.precrop_iters else None
            yield self.sampler.plan(self.size, per_image=self.per_image, crop_fraction=crop)

    def __len__(self):
        return self.end - self.start


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
