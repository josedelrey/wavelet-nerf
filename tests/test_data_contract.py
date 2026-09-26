"""Malformed inputs must fail before producing rays or silently changing pixels."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np
from PIL import Image

from modules.data import CameraRayGenerator, PixelRaySampler, camera_rays, compute_rays
from modules.datasets import load_scene
from modules.datasets.images import read_image


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pose = np.eye(4, dtype=np.float32)[None]
        self.intrinsics = np.array([[2, 0, 1], [0, 3, 1], [0, 0, 1]], dtype=np.float32)
        Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(self.root / 'view.png')

    def metadata(self, **overrides):
        value = {'camera_angle_x': .8,
                 'frames': [{'file_path': './view.png', 'transform_matrix': self.pose[0].tolist()}]}
        value.update(overrides)
        (self.root / 'transforms_train.json').write_text(json.dumps(value))

    def test_extensions_and_dot_prefixed_filenames_are_preserved(self):
        for name in ('./view.png', './view', '.hidden.png'):
            if name == '.hidden.png':
                (self.root / name).write_bytes((self.root / 'view.png').read_bytes())
            self.metadata(frames=[{'file_path': name, 'transform_matrix': self.pose[0].tolist()}])
            with self.subTest(name=name):
                scene = load_scene(self.root, splits=('train',), num_render_poses=1)
                np.testing.assert_allclose(scene.images, 128 / 255)

    def test_missing_and_malformed_split_errors_have_context(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Missing Blender split.*train'):
            load_scene(self.root, splits=('train',))
        for fields in ({'frames': []}, {'frames': {}}, {'camera_angle_x': None},
                       {'camera_angle_x': True}, {'camera_angle_x': float('nan')},
                       {'frames': [{'file_path': '', 'transform_matrix': []}]},
                       {'frames': [{'file_path': 'view.png', 'transform_matrix': [[1]]}]}):
            self.metadata(**fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                load_scene(self.root, splits=('train',))
        self.metadata(frames=[{'file_path': 'missing.png', 'transform_matrix': self.pose[0].tolist()}])
        with self.assertRaisesRegex(FileNotFoundError, 'Dataset image is missing.*missing.png'):
            load_scene(self.root, splits=('train',))

    def test_paths_cannot_escape_through_parent_or_symlink(self):
        outside = self.root.parent / f'{self.root.name}-outside.png'
        outside.write_bytes((self.root / 'view.png').read_bytes())
        self.addCleanup(outside.unlink)
        (self.root / 'linked.png').symlink_to(outside)
        for name in ('../outside.png', str(outside), 'linked.png'):
            self.metadata(frames=[{'file_path': name, 'transform_matrix': self.pose[0].tolist()}])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'dataset directory'):
                load_scene(self.root, splits=('train',))

    def test_unsupported_image_encoding_is_rejected(self):
        for mode in ('L', 'P'):
            path = self.root / f'{mode}.png'
            Image.new(mode, (2, 2)).save(path)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                read_image(path)
        # Valid RGB16 PNG: Pillow would otherwise quietly discard channel precision.
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 16, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(b'\0' + struct.pack('>HHH', 65535, 12345, 0)))
               + chunk(b'IEND', b''))
        path = self.root / 'rgb16.png'
        path.write_bytes(png)
        with self.assertRaisesRegex(ValueError, '8-bit PNG'):
            read_image(path)
        for factor in (0, 1.5, True):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                read_image(self.root / 'view.png', factor=factor)

    def test_camera_helpers_reject_bad_geometry_before_inverse(self):
        bad_pose = self.pose.copy()
        bad_pose[0, 0, 0] = np.nan
        nonrigid = self.pose.copy()
        nonrigid[0, 0, 0] = 2
        singular = self.intrinsics.copy()
        singular[:2, :2] = 2
        for poses, matrix in ((bad_pose, self.intrinsics), (nonrigid, self.intrinsics),
                              (self.pose, singular), (self.pose, -self.intrinsics),
                              (self.pose[:0], self.intrinsics)):
            with self.subTest(poses=poses, matrix=matrix):
                with self.assertRaises(ValueError):
                    camera_rays(2, 2, poses, matrix)
                with self.assertRaises(ValueError):
                    CameraRayGenerator(2, 2, poses, matrix)
        for height in (0, 2.5, True):
            with self.assertRaisesRegex(ValueError, 'positive integers'):
                camera_rays(height, 2, self.pose, self.intrinsics)

    def test_target_count_channels_range_and_dimensions(self):
        valid = np.zeros((1, 2, 2, 3), dtype=np.float32)
        for images in (np.zeros((1, 2, 2, 4), dtype=np.float32), valid.astype(np.uint8),
                       valid + np.nan, valid + 2, np.zeros((1, 0, 2, 3), dtype=np.float32),
                       np.repeat(valid, 2, axis=0)):
            with self.subTest(shape=images.shape, dtype=images.dtype), self.assertRaises(ValueError):
                compute_rays(images, self.pose, self.intrinsics)
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            PixelRaySampler(np.repeat(valid, 2, axis=0), self.pose, self.intrinsics)

    def test_sparse_and_full_rays_agree_without_target_images(self):
        poses = np.repeat(self.pose, 2, axis=0)
        matrices = np.repeat(self.intrinsics[None], 2, axis=0)
        matrices[1, 0, 0] = 4
        origins, directions = camera_rays(2, 2, poses, matrices, normalize=False)
        sparse = CameraRayGenerator(2, 2, poses, matrices, normalize=False)
        o, d = sparse.rays([0, 1], [2, 2])
        np.testing.assert_allclose(o, origins[:, 2])
        np.testing.assert_allclose(d, directions[:, 2])
        self.assertNotEqual(d[0, 0], d[1, 0])
