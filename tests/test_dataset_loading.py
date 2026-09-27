import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

import eval as evaluation
import train
from wavelet_nerf.camera import render_camera_path
from wavelet_nerf.data import camera_rays
from wavelet_nerf.datasets import load_scene
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.configuration import parse_config
from wavelet_nerf.utils import load_checkpoint


def llff_fixture(root, count=4, height=6, width=10):
    root.mkdir()
    (root / "images").mkdir()
    rows = []
    # Reverse creation order to check deterministic image/pose pairing.
    for index in reversed(range(count)):
        Image.fromarray(
            np.full((height, width, 3), (index * 40) % 256, dtype=np.uint8)
        ).save(root / "images" / f"{index:03d}.png")
    for index in range(count):
        pose = np.column_stack(
            (
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [index + 1, 2, 3],
                [height, width, 8 + index],
            )
        )
        rows.append(np.concatenate((pose.ravel(), [2, 6])))
    np.save(root / "poses_bounds.npy", np.asarray(rows, dtype=np.float64))
    return np.asarray(rows, dtype=np.float64)


def blender_fixture(root):
    root.mkdir()
    for name, count in [("train", 3), ("val", 2), ("test", 2)]:
        frames = []
        for index in range(count):
            pixels = np.zeros((6, 10, 4), dtype=np.uint8)
            pixels[..., :3] = 60
            pixels[..., 3] = 128
            path = f"{name}_{index}.png"
            Image.fromarray(pixels).save(root / path)
            pose = np.eye(4)
            pose[2, 3] = 4
            frames.append({"file_path": path, "transform_matrix": pose.tolist()})
        (root / f"transforms_{name}.json").write_text(
            json.dumps({"camera_angle_x": 0.8, "frames": frames})
        )


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_blender_explicit_splits_alpha_resize_and_intrinsics(self):
        root = self.root / "blender"
        blender_fixture(root)
        scene = load_scene(root, factor=2, num_render_poses=3, testskip=2)
        self.assertEqual(scene.dataset_type, "blender")
        self.assertEqual(scene.images.shape, (5, 3, 5, 3))
        self.assertEqual(scene.split("train").indices.tolist(), [0, 1, 2])
        self.assertEqual(scene.split("val").indices.tolist(), [0])
        self.assertEqual(scene.split("test").frame_paths, ("test_0.png",))
        np.testing.assert_allclose(scene.images, (60 / 255) * (128 / 255) + 127 / 255)
        focal = 2.5 / np.tan(0.4)
        np.testing.assert_allclose(
            scene.intrinsics[0], [[focal, 0, 2.5], [0, focal, 1.5], [0, 0, 1]]
        )
        np.testing.assert_allclose(scene.world_to_scene, np.eye(4))
        self.assertEqual(scene.render_poses.shape, (3, 4, 4))
        # Requested splits do not require JSON files for other splits.
        (root / "transforms_val.json").unlink()
        (root / "transforms_test.json").unlink()
        self.assertEqual(len(load_scene(root, splits=("train",)).images), 3)

    def test_llff_order_axes_scale_recenter_intrinsics_and_holdouts(self):
        root = self.root / "fern"
        rows = llff_fixture(root)
        scene = load_scene(root, "llff", factor=2, llff_holdout=2, num_render_poses=5)
        self.assertEqual(scene.images.shape, (4, 3, 5, 3))
        np.testing.assert_allclose(scene.images[:, 0, 0, 0], np.arange(4) * 40 / 255)
        np.testing.assert_allclose(
            scene.poses[:, :3, :3], np.broadcast_to(np.eye(3), (4, 3, 3))
        )
        np.testing.assert_allclose(scene.poses[:, :3, 3].mean(axis=0), 0, atol=1e-7)
        source_positions = rows[:, :15].reshape(-1, 3, 5)[:, :, 3]
        points = np.column_stack((source_positions, np.ones(4)))
        np.testing.assert_allclose(
            (scene.world_to_scene @ points.T).T[:, :3], scene.poses[:, :3, 3], atol=1e-6
        )
        np.testing.assert_allclose(scene.bounds, np.tile([4 / 3, 4], (4, 1)))
        np.testing.assert_allclose(scene.intrinsics[:, 0, 0], np.arange(8, 12) / 2)
        np.testing.assert_allclose(scene.intrinsics[:, 1, 1], np.arange(8, 12) / 2)
        self.assertEqual(scene.split("train").indices.tolist(), [1, 3])
        self.assertEqual(scene.split("val").indices.tolist(), [0, 2])
        self.assertEqual(scene.split("test").indices.tolist(), [0, 2])
        self.assertFalse(scene.white_background)
        self.assertEqual(scene.sampling_bounds, (0, 1))
        self.assertEqual(scene.render_path["type"], "spiral")
        self.assertTrue(np.isfinite(scene.render_poses).all())
        metadata = json.loads(json.dumps(scene.describe(), allow_nan=False))
        np.testing.assert_allclose(
            render_camera_path(metadata["render_path"], 5), scene.render_poses
        )
        self.assertEqual(
            metadata["splits"]["test"]["frame_paths"],
            ["images/000.png", "images/002.png"],
        )
        self.assertEqual(len(metadata["splits"]["train"]["intrinsics"]["matrices"]), 2)
        self.assertFalse((root / "images_2").exists())

    def test_llff_cached_images_use_actual_dimensions_without_double_resize(self):
        root = self.root / "fern"
        llff_fixture(root, height=7, width=11)
        cache = root / "images_2"
        cache.mkdir()
        for index in range(4):
            Image.fromarray(np.full((3, 6, 3), 200, dtype=np.uint8)).save(
                cache / f"{index:03d}.png"
            )
        scene = load_scene(root, "llff", factor=2, llff_recenter=False)
        self.assertEqual(scene.images.shape, (4, 3, 6, 3))
        np.testing.assert_allclose(scene.images, 200 / 255)
        self.assertAlmostEqual(float(scene.intrinsics[0, 0, 0]), 8 * 6 / 11, places=6)
        self.assertAlmostEqual(float(scene.intrinsics[0, 1, 1]), 8 * 3 / 7, places=6)
        np.testing.assert_allclose(scene.poses[0, :3, 3], np.array([1, 2, 3]) * 2 / 3)
        self.assertEqual(scene.frame_paths[0], "images_2/000.png")

    def test_reference_eighth_view_holdout_and_spiral_equations(self):
        root = self.root / "fern_protocol"
        llff_fixture(root, count=18)
        scene = load_scene(root, "llff", factor=1, llff_holdout=8)
        held_out = [0, 8, 16]
        self.assertEqual(scene.split("val").indices.tolist(), held_out)
        self.assertEqual(scene.split("test").indices.tolist(), held_out)
        self.assertEqual(
            scene.split("train").indices.tolist(),
            [index for index in range(18) if index not in held_out],
        )
        reloaded = load_scene(
            root, "llff", factor=1, llff_holdout=8, splits=("train", "val")
        )
        protocol = reloaded.describe()["split_protocol"]
        self.assertEqual(protocol["holdout_stride"], 8)
        self.assertEqual(protocol["test_indices"], held_out)
        self.assertEqual(protocol["val_indices"], held_out)
        self.assertTrue(protocol["validation_uses_test_views"])
        self.assertEqual(scene.render_poses.shape, (120, 4, 4))

        # Independent homogeneous equations from the reference spiral helper.
        average = np.asarray(scene.render_path["average_pose"])
        radii = np.r_[np.percentile(np.abs(scene.poses[:, :3, 3]), 90, axis=0), 1.0]
        close, distant = scene.bounds.min() * 0.9, scene.bounds.max() * 5
        focus_depth = 1 / (0.25 / close + 0.75 / distant)
        focus = average[:3, :4] @ [0, 0, -focus_depth, 1]
        up = scene.poses[:, :3, 1].sum(axis=0)
        up = up / np.linalg.norm(up)
        expected = []
        for theta in np.linspace(0, 4 * np.pi, 121)[:-1]:
            center = average[:3, :4] @ (
                np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * 0.5), 1])
                * radii
            )
            back = center - focus
            back /= np.linalg.norm(back)
            right = np.cross(up, back)
            right /= np.linalg.norm(right)
            pose = np.eye(4)
            pose[:3, :4] = np.column_stack((right, np.cross(back, right), back, center))
            expected.append(pose)
        np.testing.assert_allclose(scene.render_poses, expected, atol=5e-7)
        np.testing.assert_allclose(reloaded.render_poses, scene.render_poses)

    def test_camera_rays_use_each_camera_matrix(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        matrices = np.array(
            [[[2, 0, 0], [0, 4, 0], [0, 0, 1]], [[4, 0, 1], [0, 2, 1], [0, 0, 1]]]
        )
        origins, directions = camera_rays(2, 2, poses, matrices, normalize=False)
        np.testing.assert_allclose(directions[0, 3], [0.5, -0.25, -1])
        np.testing.assert_allclose(directions[1, 3], [0, 0, -1])
        self.assertTrue(origins.flags.writeable)

    def test_invalid_llff_metadata_and_image_pairing_fail_early(self):
        root = self.root / "fern"
        rows = llff_fixture(root)
        for invalid in (
            rows[:, :16],
            np.full_like(rows, np.nan),
            np.column_stack((rows[:, :15], np.zeros((4, 2)))),
        ):
            with self.subTest(shape=invalid.shape):
                np.save(root / "poses_bounds.npy", invalid)
                with self.assertRaises(ValueError):
                    load_scene(root, "llff")
        np.save(root / "poses_bounds.npy", rows)
        (root / "images" / "000.png").unlink()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            load_scene(root, "llff")

    def test_invalid_camera_geometry_and_dimensions_fail_early(self):
        root = self.root / "fern"
        rows = llff_fixture(root)
        invalid = rows.copy()
        invalid[0, 0] = 2
        np.save(root / "poses_bounds.npy", invalid)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            load_scene(root, "llff")
        np.save(root / "poses_bounds.npy", rows)
        Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8)).save(
            root / "images" / "000.png"
        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_scene(root, "llff")

    def test_invalid_loader_options(self):
        for options in (
            {"dataset_type": "unknown"},
            {"factor": 0},
            {"factor": 1.5},
            {"factor": True},
            {"splits": ()},
            {"splits": ("train", "train")},
            {"splits": ("missing",)},
            {"dataset_type": "llff", "llff_holdout": 1},
            {"dataset_type": "llff", "llff_bounds_scale": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                load_scene(self.root, **options)

    def test_fern_example_configs_resolve_for_all_models(self):
        for name in ("nerf", "siren", "wavelet"):
            config = resolve_experiment_config(
                parse_config(f"configs/config_{name}_fern.yaml")
            )
            self.assertEqual(config["dataset_type"], "llff")
            self.assertEqual(float(config["near"]), 0)
            self.assertEqual(float(config["far"]), 1)
            self.assertEqual(config["dataset_factor"], 8)
            self.assertEqual(config["white_background"], False)

    def test_llff_train_test_and_dataset_free_spiral_for_all_models(self):
        root = self.root / "fern"
        llff_fixture(root)
        for name in ("nerf", "siren", "wavelet"):
            with self.subTest(model=name), contextlib.ExitStack() as stack:
                config = {
                    "dataset_path": str(root),
                    "dataset_type": "llff",
                    "dataset_factor": 2,
                    "llff_holdout": 2,
                    "model_type": name,
                    "experiment_name": name,
                    "log_root": str(self.root / "logs"),
                    "save_path": str(self.root / "models"),
                    "num_iters": 2,
                    "num_random_rays": 4,
                    "num_samples": 4,
                    "num_samples_eval": 4,
                    "chunk_size": 8,
                    "val_interval": 1,
                    "num_render_poses": 2,
                }
                config.update(
                    {
                        "nerf": {"hidden_dim": 8, "num_importance": 4},
                        "siren": {"siren_hidden_dim": 8, "num_layers": 2},
                        "wavelet": {"wave_hidden_dim": 8, "wave_num_layers": 2},
                    }[name]
                )
                config_path = self.root / f"{name}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                loader = train.DataLoader
                stack.enter_context(
                    patch.object(
                        train,
                        "DataLoader",
                        side_effect=lambda *a, **kw: loader(
                            *a, **{**kw, "num_workers": 0}
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(torch.cuda, "is_available", return_value=False)
                )
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                stack.enter_context(patch.object(train, "SummaryWriter"))
                stack.enter_context(
                    patch("sys.argv", ["train.py", "--config", str(config_path)])
                )
                train.main()
                checkpoint_path = self.root / "models" / name / f"{name}_000002.pth"
                checkpoint = load_checkpoint(checkpoint_path)
                self.assertEqual(checkpoint["step"], 2)
                metadata = checkpoint["experiment"]["dataset"]
                self.assertEqual(metadata["type"], "llff")
                self.assertEqual(metadata["sampling_bounds"], [0, 1])
                self.assertEqual(
                    metadata["scene_normalization"],
                    {"center": [0.0, 0.0, 0.0], "scale": 1.0},
                )
                self.assertEqual(metadata["splits"]["train"]["indices"], [1, 3])
                self.assertEqual(metadata["split_protocol"]["val_indices"], [0, 2])
                self.assertEqual(metadata["split_protocol"]["test_indices"], [0, 2])
                self.assertTrue(
                    metadata["split_protocol"]["validation_uses_test_views"]
                )
                self.assertTrue(
                    all(
                        torch.isfinite(value).all()
                        for value in checkpoint["model_state_dict"].values()
                    )
                )
                output = self.root / f"{name}_test"
                with patch(
                    "sys.argv",
                    [
                        "eval.py",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--mode",
                        "test",
                        "--output",
                        str(output),
                    ],
                ):
                    evaluation.main()
                report = json.loads((output / "metrics.json").read_text())
                self.assertEqual([row["index"] for row in report["per_image"]], [0, 2])
                self.assertTrue(np.isfinite(report["summary"]["pooled_mse"]))
                with (
                    patch.object(
                        evaluation,
                        "load_configured_scene",
                        side_effect=AssertionError("dataset access"),
                    ),
                    patch(
                        "sys.argv",
                        [
                            "eval.py",
                            "--no-video",
                            "--checkpoint",
                            str(checkpoint_path),
                            "--output",
                            str(self.root / f"{name}_render"),
                        ],
                    ),
                ):
                    evaluation.main()
                self.assertTrue(
                    (self.root / f"{name}_render" / "frame_0001.png").is_file()
                )


if __name__ == "__main__":
    unittest.main()
