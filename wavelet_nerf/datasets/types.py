"""The shared, explicit scene contract used by dataset loaders and callers."""

from dataclasses import dataclass

import numpy as np
from .images import validate_rgb_images


def validate_cameras(poses, intrinsics, *, count=None):
    """Validate rigid OpenGL poses and invertible per-view pinhole intrinsics."""
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("Camera poses must be a nonempty N x 4 x 4 array")
    if count is not None and len(poses) != count:
        raise ValueError(
            f"Image/pose count mismatch: {count} images, {len(poses)} poses"
        )
    if intrinsics.shape != (len(poses), 3, 3):
        raise ValueError(
            "Camera intrinsics must have shape N x 3 x 3 matching the poses"
        )
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Camera poses and intrinsics must be finite")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1]):
        raise ValueError("Camera poses must have homogeneous bottom rows [0, 0, 0, 1]")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("Camera poses must contain proper orthonormal rotations")
    if (intrinsics[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("Camera focal lengths must be positive")
    if (
        not np.allclose(intrinsics[:, 2], [0, 0, 1])
        or (np.linalg.det(intrinsics) <= 0).any()
    ):
        raise ValueError(
            "Camera intrinsics must be invertible pinhole matrices with bottom rows [0, 0, 1]"
        )


def _describe_cameras(height, width, poses, intrinsics, bounds, indices, frame_paths):
    first = intrinsics[0]
    return {
        "indices": indices.tolist(),
        "frame_paths": list(frame_paths),
        "camera_to_world": poses.tolist(),
        "bounds": bounds.tolist(),
        "intrinsics": {
            "height": height,
            "width": width,
            "fx": float(first[0, 0]),
            "fy": float(first[1, 1]),
            "cx": float(first[0, 2]),
            "cy": float(first[1, 2]),
            "matrices": intrinsics.tolist(),
        },
    }


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
        return _describe_cameras(
            height,
            width,
            self.poses,
            self.intrinsics,
            self.bounds,
            self.indices,
            self.frame_paths,
        )


@dataclass(frozen=True)
class SceneData:
    """RGB views in one scene coordinate frame, with OpenGL camera-to-world poses.

    Intrinsics are per-view 3x3 pinhole matrices. Bounds are camera-depth bounds
    in scene units. LLFF sampling_bounds are [0, 1] in NDC. Blender uses world rays.
    world_to_scene maps source world points, separately from network normalization.
    Split indices select rows. The source_indices field preserves original dataset ordering.
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
        validate_rgb_images(self.images)
        validate_cameras(self.poses, self.intrinsics, count=count)
        for name, array, shape in (
            ("poses", self.poses, (count, 4, 4)),
            ("intrinsics", self.intrinsics, (count, 3, 3)),
            ("bounds", self.bounds, (count, 2)),
            ("world_to_scene", self.world_to_scene, (4, 4)),
        ):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Scene {name} must be finite with shape {shape}")
        if len(self.frame_paths) != count or self.source_indices.shape != (count,):
            raise ValueError(
                "Scene frame paths and source indices must match the view count"
            )
        if (self.bounds[:, 0] <= 0).any() or (
            self.bounds[:, 1] <= self.bounds[:, 0]
        ).any():
            raise ValueError("Camera bounds must satisfy 0 < near < far")
        for name, indices in self.splits.items():
            if (
                len(indices) == 0
                or indices.ndim != 1
                or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any()
                or (indices >= count).any()
            ):
                raise ValueError(
                    f"Scene split {name!r} must contain valid, nonempty view indices"
                )
        near, far = self.sampling_bounds
        if self.dataset_type == "llff":
            if (near, far) != (0.0, 1.0):
                raise ValueError("LLFF NDC sampling bounds must be (0, 1)")
        elif not np.isfinite([near, far]).all() or not 0 < near < far:
            raise ValueError("Scene sampling bounds must satisfy finite 0 < near < far")
        if (
            self.render_poses.ndim != 3
            or self.render_poses.shape[1:] != (4, 4)
            or not np.isfinite(self.render_poses).all()
        ):
            raise ValueError("Render poses must be a finite N x 4 x 4 array")

    def split(self, name):
        if name not in self.splits:
            raise ValueError(
                f"Split {name!r} was not loaded. Available splits: {', '.join(self.splits)}"
            )
        indices = self.splits[name]
        # Basic slicing shares RGB storage for contiguous Blender splits.
        selection = (
            slice(int(indices[0]), int(indices[-1]) + 1)
            if np.array_equal(indices, np.arange(indices[0], indices[-1] + 1))
            else indices
        )
        return SceneSplit(
            self.images[selection],
            self.poses[indices],
            self.intrinsics[indices],
            self.bounds[indices],
            self.source_indices[indices],
            tuple(self.frame_paths[index] for index in indices),
        )

    def describe_split(self, name):
        indices = self.splits[name]
        return _describe_cameras(
            *self.images.shape[1:3],
            self.poses[indices],
            self.intrinsics[indices],
            self.bounds[indices],
            self.source_indices[indices],
            tuple(self.frame_paths[index] for index in indices),
        )

    def describe(self):
        height, width = self.images.shape[1:3]
        return {
            "type": self.dataset_type,
            "background": "white" if self.white_background else "black",
            "ray_space": "ndc" if self.dataset_type == "llff" else "world",
            "ndc_near_plane": 1.0 if self.dataset_type == "llff" else None,
            "world_to_scene": self.world_to_scene.tolist(),
            "sampling_bounds": list(self.sampling_bounds),
            "splits": {name: self.describe_split(name) for name in self.splits},
            "render_path": self.render_path,
            **(
                {"split_protocol": self.split_protocol}
                if self.split_protocol is not None
                else {}
            ),
            "render_intrinsics": {
                "height": height,
                "width": width,
                "matrix": self.intrinsics[0].tolist(),
            },
        }
