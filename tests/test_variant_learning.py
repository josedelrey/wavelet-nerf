"""Quick CPU fits through the SIREN and wavelet rendering paths."""

import unittest

import torch
import torch.nn.functional as F

from wavelet_nerf.models import Siren, WaveletNeRF
from wavelet_nerf.rendering import render_nerf
from wavelet_nerf.scene import SceneNormalization


class VariantLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        grid = torch.linspace(-0.3, 0.3, 8)
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        x, y = x.flatten(), y.flatten()
        self.directions = F.normalize(
            torch.stack((x, y, -torch.ones_like(x)), dim=-1), dim=-1
        )
        self.origins = torch.tensor([[0.0, 0.0, 4.0]]).expand_as(self.directions)
        self.target = torch.stack((0.5 + x, 0.5 + y, 0.3 + 0.25 * x - 0.2 * y), dim=-1)

    def assert_fits_fixed_rays(self, name, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        initial = None
        for step in range(81):
            prediction = render_nerf(
                model,
                self.origins,
                self.directions,
                2.0,
                6.0,
                num_samples=8,
                device="cpu",
                white_background=True,
                chunk_size=64,
                stratified=False,
                scene_normalization=SceneNormalization(scale=2.0),
                netchunk=1024,
            )
            loss = F.mse_loss(prediction, self.target)
            self.assertTrue(torch.isfinite(loss).item(), f"{name} loss is not finite")
            if step == 0:
                initial = loss.item()
            if step == 80:
                final = loss.item()
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{name}: fixed-ray MSE {initial:.4f} -> {final:.4f}")
        self.assertLess(final, initial * 0.25, f"{name} failed to fit fixed rays")

    def test_siren_learns(self):
        torch.manual_seed(0)
        self.assert_fits_fixed_rays("SIREN", Siren(num_layers=2, hidden_dim=32))

    def test_wavelet_learns(self):
        torch.manual_seed(0)
        self.assert_fits_fixed_rays(
            "Wavelet MFN", WaveletNeRF(num_layers=2, hidden_dim=32, input_scale=16.0)
        )


if __name__ == "__main__":
    unittest.main()
