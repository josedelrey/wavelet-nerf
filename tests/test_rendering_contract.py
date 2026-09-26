"""Rendering contracts, independent sampling, and stable numeric integration."""

import unittest
import warnings

import numpy as np
import torch

from wavelet_nerf.loss import mse_to_psnr
from wavelet_nerf.models import LegacyNeRF
from wavelet_nerf.nerf_reference import raw_to_outputs, render_reference_nerf
from wavelet_nerf.rendering import (composite_volume, compute_accumulated_transmittance,
                               generate_sample_positions, render_nerf)


class Field(torch.nn.Module):
    def forward(self, points, directions, **options):
        colors = torch.sigmoid(points)
        density = torch.ones_like(points[:, :1])
        if options.get('return_raw'):
            return torch.cat((points, density), dim=-1)
        return colors, density.squeeze(-1)


class RenderingContractTests(unittest.TestCase):
    def setUp(self):
        self.origins = torch.zeros(3, 3)
        self.directions = torch.tensor([[0., 0., -1.]]).expand(3, -1)

    def test_jitter_is_independent_for_reference_and_legacy_bins(self):
        for reference in (False, True):
            torch.manual_seed(12)
            points, deltas = generate_sample_positions(self.origins, self.directions, 2., 6., 8,
                                                       reference=reference)
            depths = -points[..., 2]
            with self.subTest(reference=reference):
                self.assertFalse(torch.equal(depths[0], depths[1]))
                self.assertTrue(((depths >= 2) & (depths <= 6)).all())
                self.assertTrue((depths[:, 1:] >= depths[:, :-1]).all())
                torch.testing.assert_close(deltas[:, :-1], depths[:, 1:] - depths[:, :-1])

    def test_invalid_inputs_fail_in_both_public_renderers(self):
        for renderer in (render_nerf, render_reference_nerf):
            for options in ({'num_samples': 0}, {'num_samples': 2.5}, {'num_samples': True},
                            {'chunk_size': 0}, {'chunk_size': 1.5}, {'netchunk': -1},
                            {'num_importance': -1}, {'num_importance': 2.5},
                            {'perturb': float('nan')}, {'raw_noise_std': -1}):
                with self.subTest(renderer=renderer.__name__, options=options), self.assertRaises(ValueError):
                    renderer(Field(), self.origins, self.directions, 2, 6, **options)
            for near, far in ((6, 2), (2, 2), (-1, 6), (float('nan'), 6), (2, float('inf'))):
                with self.subTest(bounds=(near, far)), self.assertRaises(ValueError):
                    renderer(Field(), self.origins, self.directions, near, far)
            for origins, directions in ((self.origins[:0], self.directions[:0]),
                                        (self.origins, self.directions[:2]),
                                        (self.origins, torch.zeros_like(self.directions)),
                                        (self.origins, self.directions * float('nan')),
                                        (self.origins.half(), self.directions.half()),
                                        (self.origins.double(), self.directions),
                                        (self.origins[0], self.directions[0])):
                with self.subTest(shape=origins.shape), self.assertRaises(ValueError):
                    renderer(Field(), origins, directions, 2, 6)
            with self.assertRaisesRegex(ValueError, 'unit length'):
                renderer(Field(), self.origins, self.directions, 2, 6,
                         view_directions=self.directions * 2)
            with self.assertRaisesRegex(ValueError, 'near > 0'):
                renderer(Field(), self.origins, self.directions, 0, 1, lindisp=True)

    def test_hooks_run_for_research_legacy_and_reference_queries(self):
        for renderer, model in ((render_nerf, Field()),
                                (render_nerf, LegacyNeRF(hidden_dim=8)),
                                (render_reference_nerf, Field())):
            calls = []
            handle = model.register_forward_hook(lambda *args: calls.append(True))
            try:
                renderer(model, self.origins, self.directions, 2, 6, num_samples=4,
                         num_importance=2, netchunk=3, stratified=False)
                self.assertTrue(calls)
            finally:
                handle.remove()

    def test_deterministic_rendering_is_chunk_invariant(self):
        for renderer in (render_nerf, render_reference_nerf):
            results = [renderer(Field(), self.origins, self.directions, 2, 6, num_samples=4,
                                num_importance=2, stratified=False, chunk_size=chunk, netchunk=5)
                       for chunk in (1, 3)]
            if isinstance(results[0], dict):
                results = [result['rgb_map'] for result in results]
            torch.testing.assert_close(*results)

    def test_distance_scaling_is_correct_for_nonunit_geometry(self):
        for renderer in (render_nerf, render_reference_nerf):
            # Same sample positions and physical intervals after ray reparameterization.
            a = renderer(Field(), self.origins, self.directions, 2, 6,
                         num_samples=8, num_importance=0, stratified=False)
            b = renderer(Field(), self.origins, self.directions * 2, 1, 3,
                         num_samples=8, num_importance=0, stratified=False)
            if isinstance(a, dict):
                a, b = a['rgb_map'], b['rgb_map']
            torch.testing.assert_close(a, b)

    def test_dtype_is_preserved_and_half_network_outputs_integrate_in_float32(self):
        for renderer in (render_nerf, render_reference_nerf):
            result = renderer(Field(), self.origins.double(), self.directions.double(), 2, 6,
                              num_samples=4, num_importance=0, stratified=False)
            rgb = result['rgb_map'] if isinstance(result, dict) else result
            self.assertEqual(rgb.dtype, torch.float64)
        betas = torch.full((1, 3), .5, dtype=torch.float64)
        self.assertEqual(compute_accumulated_transmittance(betas).dtype, torch.float64)
        colors = torch.full((1, 3, 3), .5, dtype=torch.float16)
        densities = torch.ones(1, 3, dtype=torch.float16)
        depths = torch.tensor([[2., 3., 4.]])
        deltas = torch.tensor([[1., 1., 1e10]])
        rgb = composite_volume(colors, densities, deltas, True)
        raw = torch.cat((torch.zeros_like(colors), densities[..., None]), dim=-1)
        reference = raw_to_outputs(raw, depths, self.directions[:1])['rgb_map']
        half_geometry = raw_to_outputs(raw, depths.half(), self.directions[:1].half())['rgb_map']
        for result in (rgb, reference, half_geometry):
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(torch.isfinite(result).all())

    def test_zero_mse_has_infinite_psnr_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(mse_to_psnr(0), float('inf'))
            np.testing.assert_allclose(mse_to_psnr(np.array([0, .01, 1.])), [np.inf, 20., 0.])
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                mse_to_psnr(value)
