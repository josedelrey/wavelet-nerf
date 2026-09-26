"""The shared, explicit scene contract used by dataset loaders and callers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneSplit:
    images: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray
    bounds: np.ndarray
    indices: np.ndarray
    frame_paths: tuple[str, ...]

    def describe(self):
        height, width = self.images.shape[1:3]
        first = self.intrinsics[0]
        return {
            'indices': self.indices.tolist(),
            'frame_paths': list(self.frame_paths),
            'camera_to_world': self.poses.tolist(),
            'bounds': self.bounds.tolist(),
            'intrinsics': {
                'height': height, 'width': width,
                'fx': float(first[0, 0]), 'fy': float(first[1, 1]),
                'cx': float(first[0, 2]), 'cy': float(first[1, 2]),
                'matrices': self.intrinsics.tolist(),
            },
        }


@dataclass(frozen=True)
class SceneData:
    """RGB views in one scene coordinate frame, with OpenGL camera-to-world poses.

    Intrinsics are per-view 3x3 pinhole matrices. Bounds are camera-depth bounds
    in scene units. LLFF sampling_bounds are [0, 1] in NDC; Blender uses world rays.
    world_to_scene maps source world points, separately from network normalization.
    Split indices select rows; source_indices preserve original dataset ordering.
    """

    dataset_type: str
    images: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray
    bounds: np.ndarray
    splits: dict[str, np.ndarray]
    source_indices: np.ndarray
    frame_paths: tuple[str, ...]
    world_to_scene: np.ndarray
    sampling_bounds: tuple[float, float]
    white_background: bool
    render_poses: np.ndarray
    render_path: dict
    split_protocol: dict | None = None

    def __post_init__(self):
        count = len(self.images)
        if count == 0 or self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError('Scene images must be a nonempty N x H x W x 3 array')
        for name, array, shape in (
            ('poses', self.poses, (count, 4, 4)),
            ('intrinsics', self.intrinsics, (count, 3, 3)),
            ('bounds', self.bounds, (count, 2)),
            ('world_to_scene', self.world_to_scene, (4, 4)),
        ):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f'Scene {name} must be finite with shape {shape}')
        if not np.isfinite(self.images).all() or self.images.min() < 0 or self.images.max() > 1:
            raise ValueError('Scene images must be finite RGB in [0, 1]')
        if len(self.frame_paths) != count or self.source_indices.shape != (count,):
            raise ValueError('Scene frame paths and source indices must match the view count')
        if (self.intrinsics[:, (0, 1), (0, 1)] <= 0).any():
            raise ValueError('Camera focal lengths must be positive')
        if (self.bounds[:, 0] <= 0).any() or (self.bounds[:, 1] <= self.bounds[:, 0]).any():
            raise ValueError('Camera bounds must satisfy 0 < near < far')
        if not np.allclose(self.poses[:, 3], [0, 0, 0, 1]):
            raise ValueError('Camera poses must have homogeneous bottom rows [0, 0, 0, 1]')
        rotations = self.poses[:, :3, :3]
        if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3) \
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
            raise ValueError('Camera poses must contain proper orthonormal rotations')
        for name, indices in self.splits.items():
            if len(indices) == 0 or indices.ndim != 1 or (indices < 0).any() or (indices >= count).any():
                raise ValueError(f'Scene split {name!r} must contain valid, nonempty view indices')
        near, far = self.sampling_bounds
        if self.dataset_type == 'llff':
            if (near, far) != (0.0, 1.0):
                raise ValueError('LLFF NDC sampling bounds must be (0, 1)')
        elif not np.isfinite([near, far]).all() or not 0 < near < far:
            raise ValueError('Scene sampling bounds must satisfy finite 0 < near < far')
        if self.render_poses.ndim != 3 or self.render_poses.shape[1:] != (4, 4) \
                or not np.isfinite(self.render_poses).all():
            raise ValueError('Render poses must be a finite N x 4 x 4 array')

    def split(self, name):
        indices = self.splits[name]
        return SceneSplit(
            self.images[indices], self.poses[indices], self.intrinsics[indices],
            self.bounds[indices], self.source_indices[indices],
            tuple(self.frame_paths[index] for index in indices),
        )

    def describe(self):
        height, width = self.images.shape[1:3]
        return {
            'type': self.dataset_type,
            'background': 'white' if self.white_background else 'black',
            'ray_space': 'ndc' if self.dataset_type == 'llff' else 'world',
            'ndc_near_plane': 1.0 if self.dataset_type == 'llff' else None,
            'world_to_scene': self.world_to_scene.tolist(),
            'sampling_bounds': list(self.sampling_bounds),
            'splits': {name: self.split(name).describe() for name in self.splits},
            'render_path': self.render_path,
            **({'split_protocol': self.split_protocol} if self.split_protocol is not None else {}),
            'render_intrinsics': {'height': height, 'width': width,
                                  'matrix': self.intrinsics[0].tolist()},
        }
