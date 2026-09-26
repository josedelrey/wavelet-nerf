import unittest
from unittest.mock import patch

import torch

from wavelet_nerf.models import WaveletLayer, WaveletNeRF


class WaveletWidthTests(unittest.TestCase):
    def test_initialization_preserves_sampled_widths_and_weight_scaling(self):
        widths = torch.tensor([1e-8, 0.1, 1.0, 20.0, 100.0])
        torch.manual_seed(42)
        original_weights = torch.nn.Linear(3, 5).weight.detach().clone()
        torch.manual_seed(42)
        with patch.object(torch.distributions.Gamma, 'sample', return_value=widths):
            layer = WaveletLayer(3, 5, weight_scale=2.0)
        self.assertTrue(torch.isfinite(layer.raw_gamma).all())
        torch.testing.assert_close(layer.gamma, widths)
        torch.testing.assert_close(layer.linear.weight, original_weights * 2 * widths.sqrt()[:, None])

    def test_extreme_raw_widths_have_bounded_finite_outputs_and_gradients(self):
        for raw_width in [-1000.0, -10.0, 0.0, 1000.0]:
            with self.subTest(raw_width=raw_width):
                layer = WaveletLayer(3, 4, weight_scale=1.0)
                with torch.no_grad():
                    layer.raw_gamma.fill_(raw_width)
                points = torch.tensor([[0.0, 0.0, 0.0], [1000.0, -1000.0, 1000.0]],
                                      requires_grad=True)
                output = layer(points)
                self.assertTrue((layer.gamma > 0).all())
                self.assertTrue(torch.isfinite(layer.gamma).all())
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue((output.abs() <= 1).all())
                output.square().sum().backward()
                self.assertTrue(torch.isfinite(points.grad).all())
                for parameter in layer.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cancellation_at_filter_centers_cannot_amplify_envelopes(self):
        # Squared norms minus dot products can be slightly negative at coincident
        # points in float32. Large widths amplify that error without clamping.
        points = torch.rand(100, 3, generator=torch.Generator().manual_seed(42)) * 100
        layer = WaveletLayer(3, 100, weight_scale=1.0)
        with torch.no_grad():
            layer.mu.copy_(points)
            layer.raw_gamma.fill_(1e6)
            layer.linear.weight.zero_()
            layer.linear.bias.fill_(torch.pi / 2)
        output = layer(points)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output.abs() <= 1).all())
        output.sum().backward()
        for parameter in layer.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_optimizer_updates_cannot_make_widths_negative(self):
        layer = WaveletLayer(3, 4, weight_scale=1.0)
        optimizer = torch.optim.SGD(layer.parameters(), lr=100)
        for _ in range(5):
            optimizer.zero_grad()
            layer.gamma.sum().backward()
            self.assertTrue(torch.isfinite(layer.raw_gamma.grad).all())
            optimizer.step()
            self.assertTrue(torch.isfinite(layer.gamma).all())
            self.assertTrue((layer.gamma > 0).all())
        self.assertTrue((layer.raw_gamma < 0).all())

    def test_wavelet_nerf_backpropagates_to_raw_widths(self):
        torch.manual_seed(42)
        model = WaveletNeRF(hidden_dim=8, num_layers=2)
        points = torch.randn(4, 3, requires_grad=True)
        directions = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
        rgb, density = model(points, directions)
        self.assertTrue(torch.isfinite(rgb).all())
        self.assertTrue(torch.isfinite(density).all())
        (rgb.square().mean() + density.square().mean()).backward()
        for layer in model.base_net.filters:
            self.assertTrue((layer.gamma > 0).all())
            self.assertIsNotNone(layer.raw_gamma.grad)
            self.assertTrue(torch.isfinite(layer.raw_gamma.grad).all())
        self.assertTrue(any(layer.raw_gamma.grad.abs().sum() > 0
                            for layer in model.base_net.filters))


if __name__ == '__main__':
    unittest.main()
