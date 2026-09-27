import contextlib
import io
from itertools import product
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import eval as evaluation
from wavelet_nerf.models import NeRF, Siren, WaveletNeRF
from wavelet_nerf.rendering import normalize_positions, query_model, render_nerf
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.scene import SceneNormalization, resolve_scene_normalization
from wavelet_nerf.utils import load_checkpoint
from checkpoint_fixtures import save_test_checkpoint as save_checkpoint


class SceneNormalizationTests(unittest.TestCase):
    def test_center_cube_corners_and_outside_points(self):
        transform = SceneNormalization(center=(3, -5, 7), scale=2)
        corners = torch.tensor(list(product((-1.0, 1.0), repeat=3)))
        center = torch.tensor(transform.center)
        torch.testing.assert_close(
            normalize_positions(center[None], transform), torch.zeros(1, 3)
        )
        torch.testing.assert_close(
            normalize_positions(center + 2 * corners, transform), corners
        )
        torch.testing.assert_close(
            normalize_positions(center + torch.tensor([3.0, 0.0, 0.0]), transform),
            torch.tensor([1.5, 0.0, 0.0]),
        )

    def test_dtype_and_position_gradients_are_preserved(self):
        points = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
        normalized = normalize_positions(points, SceneNormalization(scale=2))
        self.assertEqual(normalized.dtype, points.dtype)
        normalized.sum().backward()
        torch.testing.assert_close(points.grad, torch.full_like(points, 0.5))

    def test_invalid_scene_parameters_are_rejected(self):
        for center in [(0, 0), (0, 0, float("nan")), (0, float("inf"), 0)]:
            with self.subTest(center=center), self.assertRaises(ValueError):
                SceneNormalization(center=center)
        for scale in [0, -1, float("nan"), float("inf")]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                SceneNormalization(scale=scale)

    def test_new_scene_coordinates_do_not_depend_on_sampling_bounds(self):
        config = {
            "scene_center": [1.0, -2.0, 3.0],
            "scene_scale": 4.0,
            "near": 2.0,
            "far": 6.0,
        }
        transform = resolve_scene_normalization(config)
        self.assertEqual(
            transform,
            resolve_scene_normalization({**config, "near": 0.0, "far": 100.0}),
        )
        self.assertEqual(resolve_scene_normalization({}), SceneNormalization())

    def test_checkpoint_transform_overrides_config(self):
        expected = SceneNormalization(center=(1, -2, 3), scale=4)
        restored = resolve_scene_normalization(
            {"scene_center": [99.0, 99.0, 99.0], "scene_scale": 100.0},
            {"scene_normalization": expected.to_dict()},
        )
        self.assertEqual(restored, expected)

    def test_checkpoint_round_trip_preserves_network_inputs(self):
        model = NeRF(hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        transform = SceneNormalization(center=(1, -2, 3), scale=4)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(
                0,
                model,
                optimizer,
                scheduler,
                directory,
                "nerf",
                "run",
                scene_normalization=transform,
            )
            checkpoint = load_checkpoint(path)
        restored_transform = resolve_scene_normalization({}, checkpoint)
        restored_model = NeRF(hidden_dim=8)
        restored_model.load_state_dict(checkpoint["model_state_dict"])
        points = torch.randn(5, 3)
        directions = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
        with torch.no_grad():
            expected = query_model(model, points, directions, transform)
            actual = query_model(restored_model, points, directions, restored_transform)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after)

    def test_normalization_does_not_rescale_integration_distances(self):
        class ConstantField(torch.nn.Module):
            def forward(self, points, directions):
                # Two finite unit intervals followed by an empty terminal sample.
                density = points.new_tensor([0.5, 0.5, 0.0]).repeat(len(points) // 3)
                return torch.full_like(points, 0.2), density

        origin = torch.zeros(1, 3)
        direction = torch.tensor([[0.0, 0.0, -1.0]])
        expected = torch.full((1, 3), 0.2 + 0.8 * np.exp(-1.0))
        for transform in [
            SceneNormalization(),
            SceneNormalization(center=(10, 20, 30), scale=7),
        ]:
            with self.subTest(transform=transform):
                actual = render_nerf(
                    ConstantField(),
                    origin,
                    direction,
                    1,
                    3,
                    num_samples=3,
                    stratified=False,
                    scene_normalization=transform,
                )
                torch.testing.assert_close(actual, expected)

    def test_all_models_render_and_backpropagate_with_scene_transform(self):
        for model_type in [NeRF, Siren, WaveletNeRF]:
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(42)
                model = model_type(hidden_dim=8)
                rgb = render_nerf(
                    model,
                    torch.zeros(2, 3),
                    torch.tensor([[0.0, 0.0, -1.0]]).expand(2, -1),
                    2,
                    6,
                    num_samples=4,
                    chunk_size=1,
                    stratified=False,
                    scene_normalization=SceneNormalization(scale=2),
                )
                self.assertTrue(torch.isfinite(rgb).all())
                rgb.square().mean().backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_evaluation_uses_saved_transform(self):
        transform = SceneNormalization(center=(1, -2, 3), scale=4)
        model = Siren(hidden_dim=8)
        checkpoint = {
            "model_type": "siren",
            "model_state_dict": model.state_dict(),
            "scene_normalization": transform.to_dict(),
            "experiment": {
                "config": resolve_experiment_config(
                    {"model_type": "siren", "siren_hidden_dim": 8}
                ),
                "dataset": {
                    "render_intrinsics": {
                        "height": 1,
                        "width": 1,
                        "matrix": [[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]],
                    },
                    "render_path": {"type": "orbit", "radius": 4.0, "elevation": -30.0},
                },
            },
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.ExitStack() as stack,
        ):
            config = {
                "siren_hidden_dim": 8,
                "num_render_poses": 1,
                "scene_center": [99.0, 99.0, 99.0],
                "scene_scale": 100.0,
            }
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(
                patch(
                    "sys.argv",
                    [
                        "eval.py",
                        "--no-video",
                        "--config",
                        "unused",
                        "--checkpoint",
                        "unused",
                        "--output",
                        directory,
                    ],
                )
            )
            stack.enter_context(
                patch.object(torch.cuda, "is_available", return_value=False)
            )
            stack.enter_context(
                patch.object(evaluation, "parse_config", return_value=config)
            )
            stack.enter_context(
                patch.object(evaluation, "load_checkpoint", return_value=checkpoint)
            )
            scene = SimpleNamespace(
                images=np.zeros((1, 1, 1, 3), dtype=np.float32),
                intrinsics=np.array(
                    [[[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]], dtype=np.float32
                ),
                render_path={"type": "orbit", "elevation": -30.0, "radius": 4.0},
            )
            stack.enter_context(
                patch.object(evaluation, "load_configured_scene", return_value=scene)
            )
            render = stack.enter_context(
                patch(
                    "wavelet_nerf.rendering.render_nerf", return_value=torch.zeros(1, 3)
                )
            )
            evaluation.main()
            self.assertEqual(render.call_args.kwargs["scene_normalization"], transform)
            self.assertTrue((Path(directory) / "frame_0000.png").is_file())


if __name__ == "__main__":
    unittest.main()
