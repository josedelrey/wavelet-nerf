import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import eval as evaluate
from modules.experiment import (check_dataset, describe_split, experiment_metadata,
                                resolve_experiment_config)
from modules.models import NeRF, Siren, WaveletNeRF
from modules.scene import SceneNormalization
from modules.utils import load_checkpoint, save_checkpoint


class ExperimentMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        images = np.zeros((1, 2, 2, 3), dtype=np.float32)
        poses = np.eye(4, dtype=np.float32)[None]
        split = describe_split(images, poses, 2.0)
        self.splits = {'train': split, 'val': split}

    def save(self, model_type, settings, model):
        config = resolve_experiment_config({'model_type': model_type, **settings})
        scene = SceneNormalization(scale=2)
        metadata = experiment_metadata(config, self.splits, scene)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        path = save_checkpoint(0, model, optimizer, scheduler, str(self.root),
                               model_type, model_type, scene_normalization=scene,
                               experiment=metadata)
        return Path(path)

    def test_render_restores_all_models_without_config_or_dataset(self):
        # Frequencies have identical weight shapes: output equality detects
        # restoring weights with the wrong non-parameter model settings.
        cases = [
            ('nerf', {'hidden_dim': '8', 'pos_encoding_dim': '2', 'dir_encoding_dim': '1'},
             NeRF(hidden_dim=8, pos_encoding_dim=2, dir_encoding_dim=1)),
            ('siren', {'siren_hidden_dim': '8', 'num_layers': '2', 'w0': '7',
                       'hidden_w0': '3', 'sigma_mul': '2', 'rgb_mul': '4'},
             Siren(hidden_dim=8, num_layers=2, w0=7, hidden_w0=3, sigma_mul=2, rgb_mul=4)),
            ('wavelet', {'wave_hidden_dim': '8', 'wave_num_layers': '2', 'omega0': '11',
                         'input_scale': '2', 'normalized': 'false'},
             WaveletNeRF(hidden_dim=8, num_layers=2, omega0=11, input_scale=2, normalized=False)),
        ]
        positions, directions = torch.randn(4, 3), torch.randn(4, 3)
        for name, settings, original in cases:
            with self.subTest(model=name):
                original.eval()
                path = self.save(name, {**settings, 'num_render_poses': '1'}, original)
                checkpoint = load_checkpoint(path)
                self.assertEqual(checkpoint['format_version'], 2)
                metadata = checkpoint['experiment']
                self.assertEqual(metadata['dataset']['type'], 'blender')
                self.assertEqual(metadata['dataset']['background'], 'white')
                self.assertEqual(metadata['dataset']['splits'], self.splits)
                self.assertIn('torch', metadata['environment']['packages'])
                self.assertEqual(len(metadata['environment']['uv_lock_sha256']), 64)
                self.assertIn('revision', metadata['code'])
                outputs = []

                def render(model, rays_o, *args, **kwargs):
                    with torch.no_grad():
                        outputs.append(model(positions, directions))
                    return torch.zeros(len(rays_o), 3)

                argv = ['eval.py', '--checkpoint', str(path), '--output', str(self.root / name)]
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch('sys.argv', argv), \
                     patch.object(evaluate.torch.cuda, 'is_available', return_value=False), \
                     patch.object(evaluate, 'load_configured_scene', side_effect=AssertionError('must use saved intrinsics')), \
                     patch.object(evaluate, 'render_nerf', render):
                    evaluate.main()
                self.assertEqual(len(outputs), 1)
                expected = original(positions, directions)
                for actual, reference in zip(outputs[0], expected):
                    torch.testing.assert_close(actual, reference)
                self.assertTrue((self.root / name / 'frame_0000.png').exists())

    def test_conflicting_settings_fail_and_runtime_overrides_work(self):
        config = resolve_experiment_config({'model_type': 'wavelet', 'omega0': '11'})
        checkpoint = {'experiment': {'config': config}}
        for key, value in [('omega0', '5'), ('normalized', 'false'), ('near', '1'),
                           ('model_type', 'siren'), ('learning_rate', '0.1'), ('seed', '7')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                resolve_experiment_config({key: value}, checkpoint, training=True)
        restored = resolve_experiment_config({'omega0': '11.0', 'num_iters': '200000',
                                               'num_samples_eval': '16'}, checkpoint, training=True)
        self.assertEqual(restored['num_samples_eval'], '16')
        self.assertEqual(restored['wave_hidden_dim'], '256')

    def test_changed_dataset_is_rejected(self):
        checkpoint = {'experiment': {'dataset': {'splits': self.splits}}}
        check_dataset(checkpoint, self.splits)
        altered = {'train': {**self.splits['train'], 'indices': [1]}, 'val': self.splits['val']}
        with self.assertRaisesRegex(ValueError, 'dataset'):
            check_dataset(checkpoint, altered)

    def test_legacy_warning_and_unknown_format(self):
        with self.assertWarnsRegex(UserWarning, 'Legacy checkpoint'):
            config = resolve_experiment_config({'w0': '7'}, {'model_type': 'siren'})
        self.assertEqual(config['w0'], '7')
        for payload in ({'format_version': 99}, {'format_version': 2}):
            path = self.root / 'invalid.pth'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'format'):
                load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
