"""Blender/NeRF Synthetic scene loader."""

import json
from pathlib import Path

import numpy as np

from wavelet_nerf.camera import render_camera_path
from .images import read_image, stack_images
from .types import SceneData


def load_blender_scene(dataset_path, *, factor=1, splits=('train', 'val', 'test'),
                       white_background=True, num_render_poses=80, testskip=1):
    if type(testskip) is not int or testskip < 1:
        raise ValueError('Blender testskip must be a positive integer')
    root = Path(dataset_path)
    images, poses, intrinsics, paths, source_indices = [], [], [], [], []
    split_indices = {}
    for name in splits:
        metadata_path = root / f'transforms_{name}.json'
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Missing Blender split {name!r}: {metadata_path}; '
                                    'download the split or request only available splits')
        try:
            with metadata_path.open(encoding='utf-8') as file:
                metadata = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f'Invalid Blender metadata {metadata_path}: {error}') from error
        if not isinstance(metadata, dict) or not isinstance(metadata.get('frames'), list):
            raise ValueError(f'{metadata_path}: expected a mapping with a frames list')
        frames = metadata['frames']
        if not frames:
            raise ValueError(f'Dataset split {name!r} contains no frames')
        value = metadata.get('camera_angle_x')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{metadata_path}: camera_angle_x must be a numeric angle in radians')
        angle = float(value)
        if not np.isfinite(angle) or not 0 < angle < np.pi:
            raise ValueError('camera_angle_x must be finite and between 0 and pi')
        start = len(images)
        stride = 1 if name == 'train' else testskip
        for index in range(0, len(frames), stride):
            frame = frames[index]
            if not isinstance(frame, dict) or not isinstance(frame.get('file_path'), str) \
                    or not frame['file_path'].strip() or 'transform_matrix' not in frame:
                raise ValueError(f'{metadata_path}: frame {index} requires a nonempty file_path '
                                 'and a 4 x 4 transform_matrix')
            try:
                pose = np.asarray(frame['transform_matrix'], dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(f'{metadata_path}: frame {index} transform_matrix must be numeric') from error
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f'{metadata_path}: frame {index} transform_matrix must be finite 4 x 4')
            relative = Path(frame['file_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Blender frame paths must stay inside the dataset directory')
            path = root / relative
            if not path.suffix:
                path = path.with_suffix('.png')
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'{metadata_path}: frame {index} path escapes the dataset directory')
            image, (original_height, original_width) = read_image(
                path, factor=factor, white_background=white_background, reference_blender=True,
            )
            height, width = image.shape[:2]
            focal = 0.5 * original_width / np.tan(0.5 * angle)
            intrinsics.append([[focal * width / original_width, 0, width / 2],
                               [0, focal * height / original_height, height / 2], [0, 0, 1]])
            images.append(image)
            poses.append(pose)
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
