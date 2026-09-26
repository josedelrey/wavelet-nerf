"""Strict YAML, early failures, shared construction, and checkpoint migration."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

import eval as evaluation
import train
from wavelet_nerf.configuration import parse_config
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.model_factory import create_model
from wavelet_nerf.models import LegacyNeRF, WaveletNeRF


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def read(self, content):
        path = self.root / 'config.yaml'
        path.write_text(content)
        return parse_config(path)

    def test_all_example_configs_are_valid_training_configs(self):
        configs = sorted((Path(__file__).resolve().parents[1] / 'config').rglob('*.yaml'))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(config=path.name):
                resolve_experiment_config(parse_config(path), training=True)

    def test_native_types_scientific_notation_and_literal_strings(self):
        config = self.read('''experiment_name: nerf_test
model_type: nerf
learning_rate: 5e-4
num_random_rays: 1024
white_background: false
scene_center: [1, -2.5, 3]
dataset_path: "./data/#scene=lego"
''')
        self.assertIs(type(config['learning_rate']), float)
        self.assertEqual(config['learning_rate'], .0005)
        self.assertIs(type(config['num_random_rays']), int)
        self.assertIs(config['white_background'], False)
        self.assertEqual(config['scene_center'], [1., -2.5, 3.])
        self.assertEqual(config['dataset_path'], './data/#scene=lego')

    def test_duplicate_unknown_malformed_and_unsafe_yaml_fail(self):
        cases = [
            ('num_iters: 2\nnum_iters: 3\n', 'Duplicate'),
            ('save_paht: ./models\n', 'Unknown'),
            ('scene_center: [0, 1\n', 'config.yaml'),
            ('- num_iters: 2\n', 'mapping'),
            ('', 'mapping'),
            ('num_iters = 2\n', 'mapping'),
            ('true: 2\n', 'keys must be strings'),
            ('!!python/object/apply:os.system ["echo bad"]\n', 'constructor'),
        ]
        for content, error in cases:
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, error):
                self.read(content)
        path = self.root / 'config.txt'
        path.write_text('num_iters = 2\n')
        with self.assertRaisesRegex(ValueError, 'YAML'):
            parse_config(path)

    def test_wrong_types_and_invalid_ranges_fail(self):
        cases = [
            'num_iters: "2"', 'num_iters: 2.5', 'num_iters: true',
            'seed: 012', 'seed: 1:20',
            'white_background: "false"', 'white_background: yes',
            'normalized: 1', 'first_step_render: off',
            'learning_rate: "5e-4"', 'learning_rate: .nan', 'learning_rate: .inf',
            'learning_rate: 0', 'save_interval: 0', 'val_interval: -1', 'log_interval: 0',
            'num_samples: 0', 'num_samples_eval: -1', 'chunk_size: 0', 'netchunk: 0',
            'hidden_dim: 1', 'wave_hidden_dim: 0', 'num_layers: 0', 'num_importance: -1',
            'pos_encoding_dim: -1', 'seed: 4294967296', 'scene_scale: 0',
            'scene_center: "0, 0, 0"', 'scene_center: [0, 1]', 'scene_center: [0, true, 1]',
            'scene_center: [0, .inf, 1]', 'lr_decay_factor: 0', 'precrop_frac: 1.1',
            'render_spiral_radius_scale: 0', 'render_orbit_radius: -1',
            'experiment_name: ../other', 'model_type: invalid', 'dataset_type: wrong',
        ]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.read(content + '\n')

    def test_required_fields_bounds_and_model_constraints(self):
        with self.assertRaisesRegex(ValueError, 'experiment_name'):
            resolve_experiment_config({}, training=True)
        for overrides in ({'near': 6, 'far': 2}, {'near': 0}, {'lr_min': .1},
                          {'num_samples': 2}, {'num_samples_eval': 1},
                          {'model_type': 'siren', 'wave_hidden_dim': 8}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_experiment_config({'experiment_name': 'run', **overrides}, training=True)
        for overrides in ({'near': 2}, {'far': 6}, {'white_background': True},
                          {'scene_scale': 2}, {'lindisp': True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_experiment_config({'dataset_type': 'llff', **overrides})
        config = resolve_experiment_config({'near': 'auto', 'far': 'auto'})
        self.assertEqual((config['near'], config['far']), (2., 6.))
        fern = resolve_experiment_config({'dataset_type': 'llff'})
        self.assertEqual((fern['dataset_factor'], fern['num_random_rays'],
                          fern['num_importance']), (4, 4096, 128))
        quickstart = resolve_experiment_config(parse_config('config/config_nerf_fern.yaml'))
        self.assertEqual((quickstart['dataset_factor'], quickstart['num_random_rays'],
                          quickstart['num_importance']), (8, 1024, 64))
        resumed = resolve_experiment_config({'near': 'auto', 'far': 'auto'},
                                            {'experiment': {'config': config}})
        self.assertEqual((resumed['near'], resumed['far']), (2., 6.))

    def test_invalid_training_config_precedes_data_models_and_output_creation(self):
        for overrides in ({'val_interval': 0}, {'num_samples': 2}, {'near': 7, 'far': 2},
                          {'scene_center': [0, 1]}, {'misspelled': 1}, {'experiment_name': ''}):
            config = {'experiment_name': 'run', 'dataset_path': str(self.root / 'missing'),
                      'save_path': str(self.root / 'models'), 'log_root': str(self.root / 'logs'),
                      **overrides}
            path = self.root / 'invalid.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.subTest(overrides=overrides), contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch('sys.argv', ['train.py', '--config', str(path)]))
                loader = stack.enter_context(patch.object(train, 'load_configured_scene'))
                factory = stack.enter_context(patch.object(train, 'create_model'))
                with self.assertRaises(ValueError):
                    train.main()
                loader.assert_not_called()
                factory.assert_not_called()
            self.assertFalse((self.root / 'models').exists())
            self.assertFalse((self.root / 'logs').exists())

    def test_invalid_evaluation_overrides_precede_data_and_output_creation(self):
        settings = resolve_experiment_config({'experiment_name': 'run'})
        checkpoint = {'experiment': {'config': settings}}
        path = self.root / 'invalid.yaml'
        path.write_text('num_render_poses: 0\n')
        output = self.root / 'renders'
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch('sys.argv', ['eval.py', '--checkpoint', 'unused',
                                                  '--config', str(path), '--output', str(output)]))
            stack.enter_context(patch.object(evaluation, 'load_checkpoint', return_value=checkpoint))
            loader = stack.enter_context(patch.object(evaluation, 'load_configured_scene'))
            with self.assertRaisesRegex(ValueError, 'num_render_poses'):
                evaluation.main()
            loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_shared_factory_accepts_alias_and_legacy_checkpoint_types(self):
        config = resolve_experiment_config({'model_type': 'multiscalewavelet', 'wave_hidden_dim': 8,
                                            'wave_num_layers': 2, 'normalized': False, 'omega0': 11})
        model = create_model(config)
        self.assertIsInstance(model, WaveletNeRF)
        self.assertEqual(config['model_type'], 'wavelet')
        self.assertIs(train.create_model, evaluation.create_model)
        legacy = LegacyNeRF(hidden_dim=8)
        checkpoint = {'model_type': 'nerf', 'model_state_dict': legacy.state_dict(),
                      'experiment': {'config': {'model_type': 'nerf', 'hidden_dim': '8',
                                                'half_res': 'false', 'learning_rate': '0.0005',
                                                'scene_center': '0, 0, 0', 'num_iters': '10'}}}
        restored = resolve_experiment_config({}, checkpoint)
        self.assertEqual(restored['hidden_dim'], 8)
        self.assertIs(restored['half_res'], False)
        self.assertEqual(restored['scene_center'], [0., 0., 0.])
        restored_model = create_model(restored)
        self.assertIsInstance(restored_model, LegacyNeRF)
        restored_model.load_state_dict(checkpoint['model_state_dict'])
        points, directions = torch.randn(2, 3), torch.randn(2, 3)
        for actual, expected in zip(restored_model(points, directions), legacy(points, directions)):
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
