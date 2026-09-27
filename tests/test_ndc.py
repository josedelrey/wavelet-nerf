import unittest

import numpy as np
import torch

from wavelet_nerf.data import camera_rays
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.ndc import ndc_camera_rays, project_rays_ndc
from wavelet_nerf.nerf_reference import render_reference_nerf
from wavelet_nerf.rendering import generate_sample_positions, render_nerf
from wavelet_nerf.scene import resolve_scene_normalization


class NDCTests(unittest.TestCase):
    def test_projection_matches_reference_equations_and_is_scale_invariant(self):
        origins = np.array([[0.2, -0.3, 0.4], [-0.5, 0.1, -0.2]], dtype=np.float32)
        directions = np.array([[0.3, -0.1, -1.0], [-0.2, 0.4, -1.0]], dtype=np.float32)
        height, width, focal, near = 4, 6, 3, 1
        # Literal component equations from the original NeRF helper, evaluated
        # independently of the vectorized projection implementation.
        shifted = (
            origins + (-(near + origins[:, 2]) / directions[:, 2])[:, None] * directions
        )
        expected_o = np.stack(
            (
                -1 / (width / (2 * focal)) * shifted[:, 0] / shifted[:, 2],
                -1 / (height / (2 * focal)) * shifted[:, 1] / shifted[:, 2],
                1 + 2 * near / shifted[:, 2],
            ),
            axis=-1,
        )
        expected_d = np.stack(
            (
                -1
                / (width / (2 * focal))
                * (directions[:, 0] / directions[:, 2] - shifted[:, 0] / shifted[:, 2]),
                -1
                / (height / (2 * focal))
                * (directions[:, 1] / directions[:, 2] - shifted[:, 1] / shifted[:, 2]),
                -2 * near / shifted[:, 2],
            ),
            axis=-1,
        )
        actual = project_rays_ndc(origins, directions, height, width, focal)
        for output, expected in zip(actual, (expected_o, expected_d)):
            np.testing.assert_allclose(output, expected, atol=2e-7)
        scaled = project_rays_ndc(origins, directions * 7, height, width, focal)
        for output, expected in zip(scaled, actual):
            np.testing.assert_allclose(output, expected, atol=2e-7)

    def test_camera_projection_preserves_world_viewing_directions(self):
        poses = np.eye(4, dtype=np.float32)[None]
        origins, geometry, views = ndc_camera_rays(
            4, 6, poses, np.array([[3.0, 0, 3.0], [0, 3.0, 2.0], [0, 0, 1.0]])
        )
        _, world = camera_rays(
            4, 6, poses, np.array([[3.0, 0, 3.0], [0, 3.0, 2.0], [0, 0, 1.0]])
        )
        np.testing.assert_allclose(views, world)
        np.testing.assert_allclose(np.linalg.norm(views, axis=-1), 1, atol=1e-7)
        np.testing.assert_allclose(origins[0, 0], [-1, 1, -1], atol=1e-7)
        np.testing.assert_allclose(geometry[0, 0], [0, 0, 2], atol=1e-7)
        self.assertFalse(np.allclose(np.linalg.norm(geometry, axis=-1), 1))
        np.testing.assert_allclose((origins + geometry)[..., 2], 1, atol=1e-7)

    def test_per_camera_intrinsics_preserve_unit_viewing_directions(self):
        poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        matrices = np.array(
            [[[3, 0, 3], [0, 2, 2], [0, 0, 1]], [[4, 0, 3], [0, 5, 2], [0, 0, 1]]],
            dtype=np.float32,
        )
        origins, geometry, views = ndc_camera_rays(4, 6, poses, matrices)
        self.assertEqual(origins.shape, (2, 24, 3))
        self.assertTrue(np.isfinite(geometry).all())
        np.testing.assert_allclose(np.linalg.norm(views, axis=-1), 1, atol=1e-7)
        self.assertFalse(np.allclose(views[0, 0], views[1, 0]))

    def test_ndc_projection_handles_noncentral_principal_points(self):
        poses = np.eye(4, dtype=np.float32)[None]
        matrix = np.array(
            [[4.0, 0.0, 1.0], [0.0, 3.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        origins, _, _ = ndc_camera_rays(4, 6, poses, matrix)
        expected = [
            [2 * x / 6 - 1, 1 - 2 * y / 4, -1] for y in range(4) for x in range(6)
        ]
        np.testing.assert_allclose(origins[0], expected, atol=2e-7)

    def test_generic_render_uses_world_views_and_ndc_interval_lengths(self):
        class Field(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, points, viewdirs):
                self.calls.append((points.clone(), viewdirs.clone()))
                density = points.new_tensor([0.5, 0.5, 0]).repeat(len(points) // 3)
                return torch.zeros_like(points), density

        model = Field()
        origins = torch.tensor([[-1.0, 1.0, -1.0], [0.2, 0.3, -1.0]])
        geometry = torch.tensor([[0.0, 0.0, 2.0], [0.4, -0.8, 2.0]])
        views = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        rgb = render_nerf(
            model,
            origins,
            geometry,
            0,
            1,
            num_samples=3,
            stratified=False,
            view_directions=views,
        )
        expected = torch.exp(-0.5 * geometry.norm(dim=-1))[:, None].expand(-1, 3)
        torch.testing.assert_close(rgb, expected)
        points, appearance = model.calls[0]
        torch.testing.assert_close(points.reshape(2, 3, 3)[:, -1], origins + geometry)
        torch.testing.assert_close(
            appearance.reshape(2, 3, 3), views[:, None].expand(-1, 3, -1)
        )

    def test_reference_coarse_and_fine_queries_use_world_views(self):
        class RawField(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, points, directions, **kwargs):
                self.calls.append((directions.clone(), kwargs["fine"]))
                return torch.ones(len(points), 4)

        model = RawField()
        views = torch.tensor([[0.0, 0.0, -1.0]])
        output = render_reference_nerf(
            model,
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 2.0]]),
            0,
            1,
            num_samples=4,
            num_importance=2,
            stratified=False,
            view_directions=views,
        )
        self.assertEqual([fine for _, fine in model.calls], [False, True])
        for appearance, _ in model.calls:
            torch.testing.assert_close(appearance, views.expand(len(appearance), -1))
        self.assertTrue(torch.isfinite(output["rgb_map"]).all())

    def test_stratified_sampling_is_per_ray_in_reference_midpoint_bins(self):
        torch.manual_seed(42)
        origins = torch.zeros(2, 3)
        directions = torch.tensor([[0.0, 0.0, 2.0]]).expand(2, -1)
        positions, deltas = generate_sample_positions(origins, directions, 0, 1, 4)
        depths = positions[..., 2] / 2
        self.assertFalse(torch.equal(depths[0], depths[1]))
        lower = torch.tensor([0, 1 / 6, 0.5, 5 / 6])
        upper = torch.tensor([1 / 6, 0.5, 5 / 6, 1])
        self.assertTrue(((depths >= lower) & (depths <= upper)).all())
        torch.testing.assert_close(deltas[:, :-1], depths[:, 1:] - depths[:, :-1])

    def test_llff_configuration_requires_reference_coordinate_conventions(self):
        for model_type in ("nerf", "siren", "wavelet"):
            with self.subTest(model=model_type):
                config = resolve_experiment_config(
                    {"dataset_type": "llff", "model_type": model_type}
                )
                self.assertEqual((float(config["near"]), float(config["far"])), (0, 1))
                self.assertEqual(resolve_scene_normalization(config).scale, 1)
                if model_type == "nerf":
                    self.assertEqual(int(config["num_importance"]), 128)
                    self.assertEqual(float(config["raw_noise_std"]), 1)
                    self.assertEqual(config["no_batching"], False)
                for settings in (
                    {"near": 2.0, "far": 6.0},
                    {"lindisp": True},
                    {"white_background": True},
                ):
                    with self.assertRaises(ValueError):
                        resolve_experiment_config(
                            {
                                "dataset_type": "llff",
                                "model_type": model_type,
                                **settings,
                            }
                        )
                with self.assertRaises(ValueError):
                    resolve_scene_normalization({**config, "scene_scale": 2.0})

    def test_invalid_projection_is_rejected(self):
        origins = np.zeros((1, 3))
        for directions in [np.array([[1, 0, 0]]), np.array([[0, np.nan, -1]])]:
            with self.assertRaises(ValueError):
                project_rays_ndc(origins, directions, 4, 6, 3)


if __name__ == "__main__":
    unittest.main()
