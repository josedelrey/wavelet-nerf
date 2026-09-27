"""Independent numerical checks of the original NeRF equations and architecture."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from wavelet_nerf.data import camera_rays
from wavelet_nerf.models import NeRF, NeRFMLP
from wavelet_nerf.nerf_reference import (KerasAdam, raw_to_outputs, render_reference_nerf,
                                    sample_depths, sample_pdf)
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.datasets.images import read_image
from wavelet_nerf.utils import load_checkpoint
from checkpoint_fixtures import save_test_checkpoint as save_checkpoint
from wavelet_nerf.rendering import render_nerf


class ReferenceNeRFTests(unittest.TestCase):
    def test_published_blender_and_llff_defaults_are_distinct(self):
        blender = resolve_experiment_config({'experiment_name': 'blender'}, training=True)
        fern = resolve_experiment_config({'dataset_type': 'llff', 'experiment_name': 'fern'}, training=True)
        self.assertEqual((blender['num_samples'], blender['num_importance']), (64, 128))
        self.assertEqual((blender['num_random_rays'], blender['num_iters'], blender['lr_decay']),
                         (1024, 500000, 500.))
        self.assertEqual((fern['dataset_factor'], fern['num_random_rays'], fern['num_iters'], fern['lr_decay']),
                         (4, 4096, 200000, 250.))
        self.assertEqual(fern['raw_noise_std'], 1.)
        self.assertEqual(fern['no_batching'], False)
        self.assertEqual(fern['white_background'], False)

    def test_blender_half_resolution_averages_float_rgba_before_compositing(self):
        rgba = np.array([[[255, 0, 0, 255], [0, 255, 0, 0]],
                         [[0, 0, 255, 0], [255, 255, 255, 255]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.fromarray(rgba).save(path)
            white, original_size = read_image(path, factor=2, white_background=True,
                                               reference_blender=True)
            raw_rgb, _ = read_image(path, factor=2, white_background=False, reference_blender=True)
        self.assertEqual(original_size, (2, 2))
        np.testing.assert_allclose(white, [[[.75, .75, .75]]])
        np.testing.assert_allclose(raw_rgb, [[[.5, .5, .5]]])

    def test_reference_default_queries_64_coarse_and_192_fine_points(self):
        class Field(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.queries = []

            def forward(self, points, viewdirs, *, fine, return_raw):
                self.queries.append((fine, points.clone(), viewdirs.clone()))
                raw = torch.zeros(len(points), 4)
                raw[:, 3] = 1.
                return raw

        field = Field()
        output = render_reference_nerf(field, torch.zeros(1, 3), torch.tensor([[0., 0., -2.]]),
                                        2, 6, stratified=False)
        self.assertEqual([len(query[1]) for query in field.queries], [64, 192])
        for fine, points, viewdirs in field.queries:
            torch.testing.assert_close(viewdirs, torch.tensor([[0., 0., -1.]]).expand(len(points), -1))
            self.assertTrue((-points[:, 2] >= 4).all() and (-points[:, 2] <= 12).all())
        torch.testing.assert_close(output['rgb_map'], torch.full((1, 3), .5))

    def test_mlp_matches_independent_numpy_forward(self):
        torch.manual_seed(11)
        model = NeRFMLP(2, 1, 8).double()
        self.assertEqual(len(model.layers), 8)
        self.assertEqual(model.layers[5].in_features, 8 + 15)
        self.assertEqual(model.feature.out_features, 256)
        points = np.array([[.2, -.3, .7], [-.8, .9, .1]])
        directions = np.array([[0., 0., -1.], [1., 0., 0.]])

        def encode(values, count):
            return np.concatenate([values] + [component for frequency in range(count)
                                  for component in (np.sin(2**frequency * values),
                                                    np.cos(2**frequency * values))], axis=-1)

        def dense(layer, values):
            return values @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()

        encoded = encode(points, 2)
        features = encoded
        for i, layer in enumerate(model.layers):
            features = np.maximum(dense(layer, features), 0)
            if i == 4:
                features = np.concatenate((encoded, features), axis=-1)
        sigma = dense(model.density, features)
        features = np.concatenate((dense(model.feature, features), encode(directions, 1)), axis=-1)
        rgb = dense(model.rgb, np.maximum(dense(model.view, features), 0))
        expected = np.concatenate((rgb, sigma), axis=-1)
        actual = model(torch.from_numpy(points), torch.from_numpy(directions)).detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        for layer in [*model.layers, model.feature, model.density, model.view, model.rgb]:
            self.assertEqual(torch.count_nonzero(layer.bias), 0)
            limit = math.sqrt(6 / (layer.in_features + layer.out_features))
            self.assertLessEqual(float(layer.weight.detach().abs().max()), limit)

    def test_camera_geometry_rays_are_not_normalized(self):
        pose = np.eye(4, dtype=np.float32)[None]
        pose[0, :3, 3] = [1, 2, 3]
        origins, directions = camera_rays(2, 3, pose, np.array([[2., 0, 1.5], [0, 2., 1.], [0, 0, 1.]]), normalize=False)
        expected = [[(x - 1.5) / 2, -(y - 1.) / 2, -1.] for y in range(2) for x in range(3)]
        np.testing.assert_allclose(directions[0], expected)
        np.testing.assert_allclose(origins[0], np.tile([1, 2, 3], (6, 1)))

    def test_volume_integral_matches_numpy_and_ray_length(self):
        raw = np.array([[[.1, -.2, .3, .2], [.8, .1, -.9, .5], [0., 1., .2, -1.]]])
        depths = np.array([[2., 3., 5.]])
        direction = np.array([[0., 0., -2.]])
        distances = np.concatenate((np.diff(depths), [[1e10]]), axis=-1) * 2
        alpha = 1 - np.exp(-np.maximum(raw[..., 3], 0) * distances)
        weights = alpha * np.concatenate((np.ones((1, 1)),
                                          np.cumprod(1 - alpha + 1e-10, axis=-1)[:, :-1]), axis=-1)
        rgb = (weights[..., None] / (1 + np.exp(-raw[..., :3]))).sum(axis=1)
        for white in (False, True):
            actual = raw_to_outputs(torch.from_numpy(raw), torch.from_numpy(depths),
                                    torch.from_numpy(direction), white)
            np.testing.assert_allclose(actual['weights'], weights, atol=1e-12)
            np.testing.assert_allclose(actual['rgb_map'], rgb + ((1 - weights.sum(axis=-1))[:, None]
                                                               if white else 0), atol=1e-12)
            np.testing.assert_allclose(actual['depth_map'], (weights * depths).sum(axis=-1))
        empty = raw_to_outputs(torch.zeros(1, 3, 4), torch.tensor([[2., 3., 5.]]),
                               torch.tensor([[0., 0., -1.]]), True)
        torch.testing.assert_close(empty['rgb_map'], torch.ones(1, 3))
        torch.testing.assert_close(empty['weights'], torch.zeros(1, 3))

    def test_pdf_matches_numpy_inverse_cdf_and_endpoints(self):
        bins = np.array([[.2, .7, 1.3, 2.5], [.2, .7, 1.3, 2.5]])
        weights = np.array([[0., 4., 1.], [0., 0., 0.]])
        expected = []
        for row_bins, row_weights in zip(bins, weights):
            pdf = (row_weights + 1e-5) / (row_weights + 1e-5).sum()
            cdf = np.concatenate(([0.], np.cumsum(pdf)))
            expected.append(np.interp(np.linspace(0, 1, 11), cdf, row_bins))
        actual = sample_pdf(torch.from_numpy(bins), torch.from_numpy(weights), 11, True)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        torch.testing.assert_close(actual[:, 0], torch.from_numpy(bins[:, 0]))
        torch.testing.assert_close(actual[:, -1], torch.from_numpy(bins[:, -1]))

    def test_depth_jitter_is_independent_and_bounded(self):
        torch.manual_seed(7)
        rays = torch.ones(32, 3)
        depths = sample_depths(rays, 2, 6, 5)
        lower = torch.tensor([2., 2.5, 3.5, 4.5, 5.5])
        upper = torch.tensor([2.5, 3.5, 4.5, 5.5, 6.])
        self.assertTrue(((depths >= lower) & (depths <= upper)).all())
        self.assertFalse(torch.equal(depths[0], depths[1]))
        torch.testing.assert_close(sample_depths(rays[:1], 2, 6, 5, False),
                                   torch.tensor([[2., 3., 4., 5., 6.]]))
        torch.testing.assert_close(sample_depths(rays[:1], 2, 6, 3, False, True),
                                   torch.tensor([[2., 3., 6.]]))

    def test_fine_sampling_detaches_coarse_pdf_and_both_losses_train(self):
        torch.manual_seed(1)
        model = NeRF(2, 1, 8, num_importance=6)
        for network in (model.coarse, model.fine):
            with torch.no_grad():
                network.density.bias.fill_(.2)
        origins = torch.zeros(3, 3)
        directions = torch.tensor([[0., 0., -1.], [.1, .3, -1.], [-.5, .1, -1.]])
        result = render_reference_nerf(model, origins, directions, 2, 6, 5, 6,
                                       stratified=False, chunk_size=2, netchunk=7)
        result['rgb_map'].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.fine.parameters()))
        model.zero_grad(set_to_none=True)
        result = render_reference_nerf(model, origins, directions, 2, 6, 5, 6, stratified=False)
        (result['rgb_map'].square().mean() + result['rgb0'].square().mean()).backward()
        for network in (model.coarse, model.fine):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in network.parameters()))
        with torch.no_grad():
            chunked = render_reference_nerf(model, origins, directions, 2, 6, 5, 6,
                                            stratified=False, chunk_size=1, netchunk=4)
        for key in result:
            torch.testing.assert_close(chunked[key], result[key], atol=1e-6, rtol=1e-5)

    def test_keras_adam_epsilon_hat_matches_independent_updates(self):
        parameter = torch.nn.Parameter(torch.tensor([.3, -.2], dtype=torch.float64))
        optimizer = KerasAdam([parameter], lr=.0005)
        expected = parameter.detach().numpy().copy()
        m, v = np.zeros(2), np.zeros(2)
        for step, gradient in enumerate([np.array([1e-8, -.01]), np.array([2e-8, .02]),
                                          np.array([-1e-7, .03])], start=1):
            m = .9 * m + .1 * gradient
            v = .999 * v + .001 * gradient**2
            expected -= .0005 * np.sqrt(1 - .999**step) / (1 - .9**step) * m / (np.sqrt(v) + 1e-7)
            parameter.grad = torch.from_numpy(gradient)
            optimizer.step()
            np.testing.assert_allclose(parameter.detach(), expected, atol=1e-12)

    def test_compiled_coarse_fine_checkpoint_restores_optimizer_and_next_update(self):
        torch.manual_seed(2)
        ordinary = NeRF(2, 1, 8, num_importance=4)
        for network in (ordinary.coarse, ordinary.fine):
            with torch.no_grad():
                network.density.bias.fill_(1.)
        model = torch.compile(ordinary, backend='eager')
        optimizer = KerasAdam(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .1 ** (step / 500000))

        def update(network, opt, schedule):
            opt.zero_grad()
            output = render_nerf(network, torch.zeros(2, 3),
                                 torch.tensor([[0., 0., -1.], [.2, .1, -1.]]),
                                 2, 6, num_samples=4, stratified=False, return_aux=True)
            (output['rgb_map'].square().mean() + output['rgb0'].square().mean()).backward()
            opt.step()
            schedule.step()

        update(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(1, model, optimizer, scheduler, directory, 'nerf', 'test')
            checkpoint = load_checkpoint(path)
        self.assertTrue(all(not key.startswith('_orig_mod.') for key in checkpoint['model_state_dict']))
        restored = NeRF(2, 1, 8, num_importance=4)
        restored.load_state_dict(checkpoint['model_state_dict'])
        restored_optimizer = KerasAdam(restored.parameters())
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer,
                                                               lambda step: .1 ** (step / 500000))
        restored_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restored_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        update(model, optimizer, scheduler)
        update(restored, restored_optimizer, restored_scheduler)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())



if __name__ == '__main__':
    unittest.main()
