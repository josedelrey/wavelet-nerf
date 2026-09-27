import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wavelet_nerf.data import PixelRaySampler
from wavelet_nerf.run_state import (atomic_write, capture_rng,
                               configure_reproducibility, prepare_output, restore_rng)
from checkpoint_fixtures import save_test_checkpoint as save_checkpoint


class RunStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rng = capture_rng()
        self.addCleanup(restore_rng, self.rng)
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)

    def test_all_cpu_rng_states_round_trip_with_weights_only_loading(self):
        configure_reproducibility(123, True)
        path = self.root / 'rng.pth'
        torch.save(capture_rng(), path)
        expected = random.random(), np.random.rand(3), torch.rand(3)
        restore_rng(torch.load(path, weights_only=True))
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.rand(3), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_uncommitted_pixel_sample_replays_after_restore(self):
        images = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3) / 100
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
        sampler = PixelRaySampler(images, poses, intrinsics, seed=7)
        restored = PixelRaySampler(images, poses, intrinsics, seed=99)
        first = sampler.sample(5)
        restored.load_state_dict(sampler.state_dict())
        for actual, expected in zip(restored.sample(5), first):
            np.testing.assert_array_equal(actual, expected)
        sampler.commit()
        second = sampler.sample(5)
        restored.load_state_dict(sampler.state_dict())
        for actual, expected in zip(restored.sample(5), second):
            np.testing.assert_array_equal(actual, expected)

    def test_failed_checkpoint_write_preserves_previous_file_and_removes_temp(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        path = Path(save_checkpoint(1, model, optimizer, scheduler, str(self.root), 'test', 'run'))
        previous = path.read_bytes()

        def fail(checkpoint, file):
            file.write(b'incomplete')
            raise OSError('disk full')

        with patch('wavelet_nerf.utils.torch.save', side_effect=fail), self.assertRaisesRegex(OSError, 'disk full'):
            save_checkpoint(1, model, optimizer, scheduler, str(self.root), 'test', 'run')
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_replace_failure_preserves_previous_artifact(self):
        path = self.root / 'config.yaml'
        path.write_text('previous')
        with patch('wavelet_nerf.run_state.os.replace', side_effect=OSError('replace failed')), \
             self.assertRaises(OSError):
            atomic_write(path, lambda file: file.write(b'new'))
        self.assertEqual(path.read_text(), 'previous')
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_existing_output_requires_explicit_overwrite_and_cleans_stale_frames(self):
        for name in ('frame_0000.png', 'frame_0030.png', 'test_0002.png', 'metrics.json', 'notes.txt'):
            (self.root / name).touch()
        patterns = ('frame_*.png', 'test_*.png', 'metrics.json', 'metrics.csv')
        with self.assertRaisesRegex(FileExistsError, '--overwrite'):
            prepare_output(self.root, patterns)
        prepare_output(self.root, patterns, overwrite=True)
        self.assertEqual([file.name for file in self.root.iterdir()], ['notes.txt'])


if __name__ == '__main__':
    unittest.main()
