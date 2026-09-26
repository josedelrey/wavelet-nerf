"""Blender/NeRF Synthetic scene loader."""

import json
from pathlib import Path

import numpy as np

from modules.camera import render_camera_path
from .images import read_image, stack_images
from .types import SceneData


def load_blender_scene(dataset_path, *, factor=1, splits=('train', 'val', 'test'),
                       white_background=True, num_render_poses=80, testskip=1):
    if testskip < 1:
        raise ValueError('Blender testskip must be a positive integer')
    root = Path(dataset_path)
    images, poses, intrinsics, paths, source_indices = [], [], [], [], []
    split_indices = {}
    for name in splits:
        with (root / f'transforms_{name}.json').open() as file:
            metadata = json.load(file)
        frames = metadata['frames']
        if not frames:
            raise ValueError(f'Dataset split {name!r} contains no frames')
        angle = float(metadata['camera_angle_x'])
        if not np.isfinite(angle) or not 0 < angle < np.pi:
            raise ValueError('camera_angle_x must be finite and between 0 and pi')
        start = len(images)
        stride = 1 if name == 'train' else testskip
        for index in range(0, len(frames), stride):
            frame = frames[index]
            relative = Path(frame['file_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Blender frame paths must stay inside the dataset directory')
            path = root / relative
            if not path.suffix:
                path = path.with_suffix('.png')
            image, (original_height, original_width) = read_image(
                path, factor=factor, white_background=white_background, reference_blender=True,
            )
            height, width = image.shape[:2]
            focal = 0.5 * original_width / np.tan(0.5 * angle)
            intrinsics.append([[focal * width / original_width, 0, width / 2],
                               [0, focal * height / original_height, height / 2], [0, 0, 1]])
            images.append(image)
            poses.append(frame['transform_matrix'])
            paths.append(frame['file_path'])
            source_indices.append(index)
        split_indices[name] = np.arange(start, len(images))
    path = {'type': 'orbit', 'elevation': -30.0, 'radius': 4.0}
    return SceneData(
        'blender', stack_images(images), np.asarray(poses, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float32),
        np.tile(np.array([2, 6], dtype=np.float32), (len(images), 1)),
        split_indices, np.asarray(source_indices), tuple(paths), np.eye(4, dtype=np.float32),
        (2.0, 6.0), white_background, render_camera_path(path, num_render_poses), path,
    )
