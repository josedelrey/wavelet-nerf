"""Forward-facing LLFF scenes, including Fern.

Camera conversion, bound scaling, recentering, and spiral geometry follow
bmild/nerf/load_llff.py (MIT, notice preserved in LICENSE).
"""

from pathlib import Path

import numpy as np

from wavelet_nerf.camera import average_pose, render_camera_path
from .images import read_image, stack_images
from .types import SceneData


def load_llff_scene(
    dataset_path,
    *,
    factor=8,
    holdout=8,
    bounds_scale=0.75,
    recenter=True,
    splits=("train", "val", "test"),
    white_background=False,
    num_render_poses=120,
):
    if type(holdout) is not int or holdout < 2:
        raise ValueError("LLFF holdout interval must be at least 2")
    if not np.isfinite(bounds_scale) or bounds_scale <= 0:
        raise ValueError("LLFF bounds_scale must be finite and positive")
    root = Path(dataset_path)
    pose_path = root / "poses_bounds.npy"
    if not pose_path.is_file():
        raise FileNotFoundError(
            f"Missing LLFF camera metadata: {pose_path}. Download poses_bounds.npy with the images"
        )
    rows = np.load(pose_path, allow_pickle=False)
    if (
        rows.ndim != 2
        or rows.shape[1] != 17
        or len(rows) < 2
        or not np.issubdtype(rows.dtype, np.number)
        or not np.isrealobj(rows)
        or not np.isfinite(rows).all()
    ):
        raise ValueError(
            "poses_bounds.npy must be a finite N x 17 array with at least two views"
        )
    source_poses = rows[:, :15].reshape(-1, 3, 5).astype(np.float64)
    bounds = rows[:, -2:].astype(np.float64)
    if (bounds[:, 0] <= 0).any() or (bounds[:, 1] <= bounds[:, 0]).any():
        raise ValueError("LLFF bounds must satisfy 0 < near < far")
    hwf = source_poses[:, :, 4]
    if (hwf <= 0).any():
        raise ValueError("LLFF height, width, and focal length must be positive")

    cached = root / f"images_{factor}"
    image_dir = cached if factor != 1 and cached.is_dir() else root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Missing LLFF image directory: {image_dir}. "
            f"provide images/ or images_{factor}/ with RGB PNG/JPEG files"
        )
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(paths) != len(rows):
        raise ValueError(
            f"LLFF image/pose count mismatch: {len(paths)} images, {len(rows)} poses"
        )
    images, intrinsics = [], []
    for index, path in enumerate(paths):
        image, original_shape = read_image(
            path,
            factor=1 if image_dir == cached else factor,
            white_background=white_background,
        )
        original_height, original_width, focal = hwf[index]
        if image_dir != cached and not np.allclose(original_shape, hwf[index, :2]):
            raise ValueError(
                f"LLFF image dimensions disagree with pose metadata: {path}"
            )
        height, width = image.shape[:2]
        intrinsics.append(
            [
                [focal * width / original_width, 0, width / 2],
                [0, focal * height / original_height, height / 2],
                [0, 0, 1],
            ]
        )
        images.append(image)

    # LLFF stores [down, right, back]. The renderer uses [right, up, back].
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :4] = np.stack(
        (
            source_poses[:, :, 1],
            -source_poses[:, :, 0],
            source_poses[:, :, 2],
            source_poses[:, :, 3],
        ),
        axis=-1,
    )
    scale = 1 / (bounds.min() * bounds_scale)
    poses[:, :3, 3] *= scale
    bounds *= scale
    world_to_scene = np.diag([scale, scale, scale, 1.0])
    if recenter:
        transform = np.linalg.inv(average_pose(poses))
        poses = transform @ poses
        world_to_scene = transform @ world_to_scene

    test = np.arange(0, len(rows), holdout)
    train = np.setdiff1d(np.arange(len(rows)), test)
    # Match the reference LLFF protocol: validation and test share held-out views.
    split_indices = {
        name: {"train": train, "val": test, "test": test}[name] for name in splits
    }
    average = average_pose(poses)
    close, distant = bounds.min() * 0.9, bounds.max() * 5
    path = {
        "type": "spiral",
        "average_pose": average.tolist(),
        "up": poses[:, :3, 1].sum(axis=0).tolist(),
        "radii": np.percentile(np.abs(poses[:, :3, 3]), 90, axis=0).tolist(),
        "focus_depth": float(1 / (0.25 / close + 0.75 / distant)),
        "rotations": 2,
        "zrate": 0.5,
    }
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    return SceneData(
        "llff",
        stack_images(images),
        poses.astype(np.float32),
        intrinsics,
        bounds.astype(np.float32),
        split_indices,
        np.arange(len(rows)),
        tuple(path.relative_to(root).as_posix() for path in paths),
        world_to_scene.astype(np.float32),
        (0.0, 1.0),
        white_background,
        render_camera_path(path, num_render_poses),
        path,
        split_protocol={
            "type": "llff_every_nth",
            "holdout_stride": holdout,
            "start_index": 0,
            "ordering": "lexicographically sorted image filenames",
            "train_indices": train.tolist(),
            "val_indices": test.tolist(),
            "test_indices": test.tolist(),
            "validation_uses_test_views": True,
        },
    )
