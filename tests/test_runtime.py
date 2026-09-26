import unittest
from unittest.mock import patch

import torch

from modules.configuration import normalize_config
from modules.experiment import resolve_experiment_config
from modules.models import NeRF, LegacyNeRF, Siren, WaveletNeRF
from modules.rendering import render_nerf
from modules.runtime import prepare_model, resolve_device


class RuntimeTests(unittest.TestCase):
    def test_device_selection_and_unavailable_cuda_errors(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto'), torch.device('cpu'))
            self.assertEqual(resolve_device('cpu'), torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                resolve_device('cuda')
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.set_device') as select:
            self.assertEqual(resolve_device('cpu'), torch.device('cpu'))
            self.assertEqual(resolve_device('auto'), torch.device('cuda'))
            self.assertEqual(resolve_device('cuda:1'), torch.device('cuda:1'))
            select.assert_called_once_with(torch.device('cuda:1'))
            with self.assertRaisesRegex(ValueError, 'index'):
                resolve_device('cuda:2')
            with self.assertRaisesRegex(ValueError, 'index'):
                resolve_device('cuda:256')

    def test_compile_is_opt_in_on_both_devices(self):
        model = torch.nn.Linear(3, 3)
        with patch('torch.compile', return_value='compiled') as compile:
            for device in (torch.device('cpu'), torch.device('cuda')):
                self.assertIs(prepare_model(model, {'compile_model': False}, device), model)
            compile.assert_not_called()
            self.assertEqual(prepare_model(model, {'compile_model': True}, torch.device('cpu')), 'compiled')
            compile.assert_called_with(model, backend='inductor', mode='default')
            prepare_model(model, {'compile_model': True}, torch.device('cuda'))
            compile.assert_called_with(model, backend='inductor', mode='reduce-overhead')

    def test_runtime_config_validation_and_checkpoint_overrides(self):
        for settings in ({'device': 'mps'}, {'device': 'cpu:1'}, {'device': 'cuda:-1'},
                         {'num_workers': -1}, {'num_workers': True}, {'compile_model': 'true'},
                         {'netchunk': 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                normalize_config(settings)
        saved = resolve_experiment_config({'device': 'cuda:1', 'compile_model': True,
                                            'num_workers': 4, 'netchunk': 100, 'chunk_size': 100})
        checkpoint = {'experiment': {'config': saved}}
        defaults = resolve_experiment_config({}, checkpoint)
        self.assertEqual(defaults['device'], 'auto')
        self.assertFalse(defaults['compile_model'])
        self.assertEqual(defaults['num_workers'], 0)
        self.assertEqual(defaults['netchunk'], 100)
        self.assertEqual(defaults['chunk_size'], 100)
        overrides = resolve_experiment_config({'device': 'cpu', 'netchunk': 3,
                                                'num_workers': 1}, checkpoint)
        self.assertEqual(overrides['device'], 'cpu')
        self.assertEqual(overrides['netchunk'], 3)
        self.assertEqual(overrides['num_workers'], 1)

    def test_query_chunking_preserves_outputs_and_gradients_for_all_models(self):
        torch.manual_seed(7)
        models = [NeRF(hidden_dim=8, pos_encoding_dim=2, dir_encoding_dim=1, num_importance=2),
                  LegacyNeRF(hidden_dim=8, pos_encoding_dim=2, dir_encoding_dim=1),
                  Siren(hidden_dim=8, num_layers=2),
                  WaveletNeRF(hidden_dim=8, num_layers=2, input_scale=2)]
        origins = torch.zeros(2, 3)
        directions = torch.tensor([[0., 0., -1.], [.2, .3, -1.]])
        for model in models:
            with self.subTest(model=type(model).__name__):
                expected = render_nerf(model, origins, directions, 2, 6,
                                       num_samples=4, stratified=False, netchunk=1000)
                expected.sum().backward()
                gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
                model.zero_grad(set_to_none=True)
                calls = []
                hook = model.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
                self.addCleanup(hook.remove)
                actual = render_nerf(model, origins, directions, 2, 6,
                                     num_samples=4, stratified=False, netchunk=3)
                actual.sum().backward()
                self.assertGreater(len(calls), 1)
                self.assertLessEqual(max(calls), 3)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                for parameter, gradient in zip(model.parameters(), gradients):
                    if gradient is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, gradient, atol=1e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
