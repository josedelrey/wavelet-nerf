"""Checkpoint destination compatibility and collision-free example names."""
from pathlib import Path
import unittest

from modules.experiment import resolve_experiment_config
from modules.utils import parse_config


class OutputConfigTests(unittest.TestCase):
    def test_conflicting_directory_names_fail(self):
        with self.assertRaisesRegex(ValueError, 'save_root and save_path'):
            resolve_experiment_config({'save_path': './one', 'save_root': './two'})
        for key in ('save_path', 'save_root'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'nonempty'):
                resolve_experiment_config({key: ''})
        resolved = resolve_experiment_config({'save_path': './models', 'save_root': 'models/.'})
        self.assertEqual(resolved['save_path'], './models')
        self.assertNotIn('save_root', resolved)

    def test_old_saved_settings_are_canonicalized_without_mutation(self):
        for saved, expected in (
            ({'save_root': './old'}, './old'),
            ({'save_root': './ignored', 'save_path': './actual'}, './actual'),
        ):
            checkpoint = {'experiment': {'config': {'model_type': 'nerf', **saved}}}
            restored = resolve_experiment_config({}, checkpoint)
            self.assertEqual(restored['save_path'], expected)
            self.assertNotIn('save_root', restored)
            self.assertEqual(checkpoint['experiment']['config'], {'model_type': 'nerf', **saved})
            moved = resolve_experiment_config({'save_root': './relocated'}, checkpoint)
            self.assertEqual(moved['save_path'], './relocated')

    def test_example_configs_use_canonical_directory_and_distinct_scene_names(self):
        names = []
        for path in sorted(Path('config').glob('config_*.txt')):
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
