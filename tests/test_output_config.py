"""Checkpoint destination compatibility and collision-free example names."""
from pathlib import Path
import unittest

from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.configuration import parse_config


class OutputConfigTests(unittest.TestCase):
    def test_removed_alias_is_rejected_and_empty_path_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unknown configuration keys'):
            resolve_experiment_config({'save_root': './models'})
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            resolve_experiment_config({'save_path': ''})

    def test_example_configs_use_canonical_directory_and_distinct_scene_names(self):
        names = []
        for path in sorted(Path('config').glob('config_*.yaml')):
            with self.subTest(config=path.name):
                config = parse_config(path)
                self.assertIn('save_path', config)
                self.assertNotIn('save_root', config)
                scene = 'fern' if 'fern' in path.stem else 'lego'
                self.assertIn(scene, config['experiment_name'])
                names.append(config['experiment_name'])
        self.assertEqual(len(names), len(set(names)))


if __name__ == '__main__':
    unittest.main()
