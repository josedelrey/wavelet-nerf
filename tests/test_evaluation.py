from checkpoint_fixtures import dataset_metadata, save_test_checkpoint
import contextlib
import csv
import io
import json
import math
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

import eval as evaluation
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.models import NeRF
from wavelet_nerf.scene import SceneNormalization


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_latest_checkpoint_uses_numeric_steps_and_ignores_other_files(self):
        directory = self.root / "models" / "example"
        directory.mkdir(parents=True)
        config = {"experiment_name": "example", "save_path": str(directory.parent)}
        with self.assertRaisesRegex(FileNotFoundError, "train this experiment first"):
            evaluation.latest_checkpoint(config)
        for name in (
            "example_9.pth",
            "example_10.pth",
            "example_bad.pth",
            "other_99.pth",
        ):
            (directory / name).touch()
        (directory / "example_100.pth").mkdir()
        self.assertEqual(
            evaluation.latest_checkpoint(config), directory / "example_10.pth"
        )

    def test_config_only_render_selects_latest_checkpoint_and_reference_outputs(self):
        for kind in ("blender", "llff"):
            with self.subTest(dataset=kind):
                name = f"example_{kind}"
                directory = self.root / "models" / name
                directory.mkdir(parents=True)
                config = resolve_experiment_config(
                    {
                        "experiment_name": name,
                        "dataset_type": kind,
                        "dataset_path": str(self.root / "missing_dataset"),
                        "save_path": str(directory.parent),
                        "log_root": str(self.root / "logs"),
                        "hidden_dim": 8,
                        "pos_encoding_dim": 2,
                        "dir_encoding_dim": 1,
                        "num_importance": 2,
                        "num_samples_eval": 4,
                        "num_render_poses": 2,
                        "device": "cpu",
                    }
                )
                model = NeRF(
                    hidden_dim=8,
                    pos_encoding_dim=2,
                    dir_encoding_dim=1,
                    num_importance=2,
                )
                optimizer = torch.optim.Adam(model.parameters())
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                metadata = dataset_metadata()
                if kind == "llff":
                    metadata["render_path"] = {
                        "type": "spiral",
                        "average_pose": np.eye(4).tolist(),
                        "up": [0, 1, 0],
                        "radii": [1, 1, 0.2],
                        "focus_depth": 5,
                        "rotations": 2,
                        "zrate": 0.5,
                    }
                for step in (1, 2):
                    checkpoint = save_test_checkpoint(
                        step,
                        model,
                        optimizer,
                        scheduler,
                        directory,
                        "nerf",
                        name,
                        experiment={"config": config, "dataset": metadata},
                    )
                path = self.root / f"{name}.yaml"
                path.write_text(yaml.safe_dump(config))
                output = self.root / "logs" / name / "renderonly_path_000002"
                video = output.parent / f"{name}_spiral_000002_rgb.mp4"
                with (
                    patch("sys.argv", ["eval.py", "--config", str(path)]),
                    patch.object(
                        evaluation, "load_checkpoint", wraps=evaluation.load_checkpoint
                    ) as load,
                    patch.object(
                        evaluation,
                        "load_configured_scene",
                        side_effect=AssertionError("dataset access"),
                    ),
                    patch.object(
                        evaluation.shutil, "which", return_value="/usr/bin/ffmpeg"
                    ),
                    patch.object(evaluation, "write_video") as write_video,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    evaluation.main()
                    load.assert_called_once_with(Path(checkpoint))
                    write_video.assert_called_once_with(
                        output, "%03d.png", video, overwrite=False
                    )
                    self.assertTrue((output / "000.png").is_file())
                    self.assertTrue((output / "001.png").is_file())
                    with self.assertRaises(FileExistsError):
                        evaluation.main()
                    with patch(
                        "sys.argv", ["eval.py", "--config", str(path), "--overwrite"]
                    ):
                        evaluation.main()
                    self.assertTrue(write_video.call_args.kwargs["overwrite"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_video_encodes_odd_sized_frames(self):
        for index in range(2):
            imageio.imwrite(
                self.root / f"frame_{index:04d}.png",
                np.full((3, 5, 3), index * 100, dtype=np.uint8),
            )
        video = self.root / "video.mp4"
        evaluation.write_video(self.root, "frame_%04d.png", video)
        self.assertGreater(video.stat().st_size, 0)
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(len(decoded.stdout), 2 * 4 * 6 * 3)

    def test_known_psnr_and_invalid_images(self):
        target = np.zeros((2, 3, 3), dtype=np.float32)
        result = evaluation.image_metrics(np.full_like(target, 0.5), target)
        self.assertEqual(result["mse"], 0.25)
        self.assertAlmostEqual(result["psnr_db"], -10 * math.log10(0.25))
        self.assertTrue(math.isinf(evaluation.image_metrics(target, target)["psnr_db"]))
        for prediction in (target[:1], np.full_like(target, np.nan)):
            with self.assertRaises(ValueError):
                evaluation.image_metrics(prediction, target)
        with self.assertRaises(ValueError):
            evaluation.image_metrics(target[:0], target[:0])

    def test_aggregation_distinguishes_mean_psnr_from_pooled_error(self):
        rows = [
            {
                "index": 0,
                "height": 1,
                "width": 1,
                "mse": 0.25,
                "psnr_db": -10 * math.log10(0.25),
            },
            {
                "index": 1,
                "height": 1,
                "width": 3,
                "mse": 0.0625,
                "psnr_db": -10 * math.log10(0.0625),
            },
        ]
        summary = evaluation.write_metrics(rows, self.root, {})
        self.assertAlmostEqual(summary["mean_psnr_db"], 9.0308998699)
        self.assertEqual(summary["pooled_mse"], (0.25 + 3 * 0.0625) / 4)
        self.assertAlmostEqual(
            summary["pooled_psnr_db"], -10 * math.log10(summary["pooled_mse"])
        )
        self.assertNotAlmostEqual(summary["mean_psnr_db"], summary["pooled_psnr_db"])
        with self.assertRaises(ValueError):
            evaluation.write_metrics([], self.root, {})

    def test_exact_match_uses_standard_json_without_capping_psnr(self):
        rows = [{"index": 0, "height": 1, "width": 1, "mse": 0.0, "psnr_db": math.inf}]
        evaluation.write_metrics(rows, self.root, {})

        def invalid_constant(value):
            self.fail(f"Nonstandard JSON constant: {value}")

        report = json.loads(
            (self.root / "metrics.json").read_text(), parse_constant=invalid_constant
        )
        self.assertEqual(report["summary"]["mean_psnr_db"], "Infinity")
        self.assertEqual(report["summary"]["pooled_psnr_db"], "Infinity")
        self.assertEqual(report["per_image"][0]["mse"], 0)

    def test_test_mode_uses_all_test_cameras_intrinsics_and_rgba_targets(self):
        scene = self.root / "relocated_scene"
        scene.mkdir()
        frames = []
        for index, alpha in enumerate((255, 0)):
            rgba = np.zeros((2, 3, 4), dtype=np.uint8)
            rgba[..., 3] = alpha
            imageio.imwrite(scene / f"{index}.png", rgba)
            pose = np.eye(4)
            pose[0, 3] = index + 1
            frames.append(
                {"file_path": f"./{index}", "transform_matrix": pose.tolist()}
            )
        (scene / "transforms_test.json").write_text(
            json.dumps({"camera_angle_x": 0.8, "frames": frames})
        )
        config = resolve_experiment_config(
            {
                "hidden_dim": 8,
                "num_samples_eval": 4,
                "chunk_size": 2,
                "dataset_path": "missing_original_dataset",
            }
        )
        model = NeRF(hidden_dim=8)
        from wavelet_nerf.experiment import experiment_metadata

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        metadata = experiment_metadata(
            config, {}, SceneNormalization(), dataset_metadata()
        )
        checkpoint = save_test_checkpoint(
            0,
            model,
            optimizer,
            scheduler,
            self.root,
            "nerf",
            "model",
            experiment=metadata,
        )
        seen = []

        def render(model, origins, directions, *args, **kwargs):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(kwargs["stratified"])
            self.assertTrue(kwargs["white_background"])
            self.assertLessEqual(len(origins), 2)
            self.assertEqual(kwargs["output_device"], "cpu")
            seen.append((origins.clone(), directions.clone()))
            return torch.full((len(origins), 3), float(origins[0, 0]) / 4)

        output = self.root / "evaluation"
        argv = [
            "eval.py",
            "--mode",
            "test",
            "--checkpoint",
            str(checkpoint),
            "--dataset-path",
            str(scene),
            "--output",
            str(output),
        ]
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            patch("sys.argv", argv),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                evaluation,
                "render_camera_path",
                side_effect=AssertionError("must use test poses"),
            ),
            patch("wavelet_nerf.rendering.render_nerf", render),
        ):
            evaluation.main()
        self.assertEqual(len(seen), 6)
        seen = [
            (
                torch.cat([item[0] for item in seen[start : start + 3]]),
                torch.cat([item[1] for item in seen[start : start + 3]]),
            )
            for start in (0, 3)
        ]
        self.assertEqual(len(seen), 2)
        self.assertEqual(float(seen[0][0][0, 0]), 1)
        self.assertEqual(float(seen[1][0][0, 0]), 2)
        focal = 1.5 / math.tan(0.4)
        expected_direction = torch.tensor([-1.5, 1.0, -focal])
        expected_direction /= focal
        torch.testing.assert_close(seen[0][1][0], expected_direction)
        report = json.loads((output / "metrics.json").read_text())
        self.assertEqual(report["summary"]["num_images"], 2)
        self.assertEqual(
            [row["file_path"] for row in report["per_image"]], ["./0", "./1"]
        )
        self.assertEqual([row["mse"] for row in report["per_image"]], [0.0625, 0.25])
        self.assertAlmostEqual(report["summary"]["mean_psnr_db"], 9.0308998699)
        with (output / "metrics.csv").open() as file:
            self.assertEqual(len(list(csv.DictReader(file))), 2)
        self.assertTrue((output / "test_0000.png").exists())
        self.assertTrue((output / "test_0001.png").exists())
        self.assertFalse((output / "frame_0000.png").exists())


if __name__ == "__main__":
    unittest.main()
