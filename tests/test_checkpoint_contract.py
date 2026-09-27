"""Only complete current-format checkpoints are accepted."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from checkpoint_fixtures import save_test_checkpoint
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.scene import resolve_scene_normalization
from wavelet_nerf.utils import CHECKPOINT_FORMAT_VERSION, load_checkpoint


class CheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        self.path = save_test_checkpoint(0, model, optimizer, scheduler, self.root, 'nerf', 'complete')
        self.checkpoint = load_checkpoint(self.path)

    def reject(self, payload):
        torch.save(payload, self.root / 'invalid.pth')
        with self.assertRaises(ValueError):
            load_checkpoint(self.root / 'invalid.pth')

    def test_only_current_format_and_fields_are_accepted(self):
        for version in (None, 1, 2, 99):
            payload = deepcopy(self.checkpoint)
            payload['format_version'] = version
            with self.subTest(version=version):
                self.reject(payload)
        for key in self.checkpoint:
            payload = deepcopy(self.checkpoint)
            del payload[key]
            with self.subTest(missing=key):
                self.reject(payload)
        self.reject({**self.checkpoint, 'step_semantics': 'completed_updates'})
        self.reject({**self.checkpoint, 'step': -1})
        self.assertEqual(self.checkpoint['format_version'], CHECKPOINT_FORMAT_VERSION)

    def test_camera_and_training_state_are_required(self):
        for category, fields in (('dataset', ('render_intrinsics', 'render_path', 'world_to_scene', 'splits')),
                                 ('training_state', ('rng', 'sampler')),
                                 ('scene_normalization', ('center', 'scale'))):
            for field in fields:
                payload = deepcopy(self.checkpoint)
                mapping = payload['experiment']['dataset'] if category == 'dataset' else payload[category]
                del mapping[field]
                with self.subTest(category=category, missing=field):
                    self.reject(payload)
        with self.assertRaises(KeyError):
            resolve_scene_normalization({}, {})

    def test_weight_names_are_never_rewritten(self):
        payload = deepcopy(self.checkpoint)
        payload['model_state_dict'] = {'_orig_mod.' + name: value for name, value in
                                       payload['model_state_dict'].items()}
        self.reject(payload)

    def test_saved_configuration_is_native_and_complete(self):
        for field, value in (('hidden_dim', '8'), ('half_res', 'false'), ('model_type', 'multiscalewavelet')):
            payload = deepcopy(self.checkpoint)
            payload['experiment']['config'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_experiment_config({}, payload)
        payload = deepcopy(self.checkpoint)
        del payload['experiment']['config']['hidden_dim']
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            resolve_experiment_config({}, payload)
