from checkpoint_fixtures import dataset_metadata
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import eval as evaluate
from wavelet_nerf.experiment import (
    check_dataset,
    experiment_metadata,
    resolve_experiment_config,
    accelerator_metadata,
)
from wavelet_nerf.models import NeRF, Siren, WaveletNeRF
from wavelet_nerf.scene import SceneNormalization
from wavelet_nerf.utils import load_checkpoint
from checkpoint_fixtures import save_test_checkpoint as save_checkpoint


class ExperimentMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        images = np.zeros((1, 2, 2, 3), dtype=np.float32)
        poses = np.eye(4, dtype=np.float32)[None]
        from wavelet_nerf.datasets.types import SceneSplit

        split = SceneSplit(
            images,
            poses,
            np.array([[[2.0, 0, 1], [0, 2.0, 1], [0, 0, 1]]]),
            np.array([[2.0, 6.0]]),
            np.array([0]),
            ("view.png",),
        ).describe()
        self.splits = {"train": split, "val": split}

    def save(self, model_type, settings, model):
        config = resolve_experiment_config({"model_type": model_type, **settings})
        scene = SceneNormalization(scale=1 if model_type == "nerf" else 2)
        metadata = experiment_metadata(config, self.splits, scene, dataset_metadata())
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        path = save_checkpoint(
            0,
            model,
            optimizer,
            scheduler,
            str(self.root),
            model_type,
            model_type,
            scene_normalization=scene,
            experiment=metadata,
        )
        return Path(path)

    def test_render_restores_all_models_without_config_or_dataset(self):
        # Frequencies have identical weight shapes: output equality detects
        # restoring weights with the wrong non-parameter model settings.
        cases = [
            (
                "nerf",
                {"hidden_dim": 8, "pos_encoding_dim": 2, "dir_encoding_dim": 1},
                NeRF(hidden_dim=8, pos_encoding_dim=2, dir_encoding_dim=1),
            ),
            (
                "siren",
                {
                    "siren_hidden_dim": 8,
                    "num_layers": 2,
                    "w0": 7.0,
                    "hidden_w0": 3.0,
                    "sigma_mul": 2.0,
                    "rgb_mul": 4.0,
                },
                Siren(
                    hidden_dim=8,
                    num_layers=2,
                    w0=7,
                    hidden_w0=3,
                    sigma_mul=2,
                    rgb_mul=4,
                ),
            ),
            (
                "wavelet",
                {
                    "wave_hidden_dim": 8,
                    "wave_num_layers": 2,
                    "omega0": 11.0,
                    "input_scale": 2.0,
                    "normalized": False,
                },
                WaveletNeRF(
                    hidden_dim=8,
                    num_layers=2,
                    omega0=11,
                    input_scale=2,
                    normalized=False,
                ),
            ),
        ]
        positions, directions = torch.randn(4, 3), torch.randn(4, 3)
        for name, settings, original in cases:
            with self.subTest(model=name):
                original.eval()
                path = self.save(name, {**settings, "num_render_poses": 1}, original)
                checkpoint = load_checkpoint(path)
                self.assertEqual(checkpoint["format_version"], 3)
                metadata = checkpoint["experiment"]
                self.assertEqual(metadata["dataset"]["type"], "blender")
                self.assertEqual(metadata["dataset"]["background"], "white")
                self.assertEqual(metadata["dataset"]["splits"], self.splits)
                self.assertIn("torch", metadata["environment"]["packages"])
                self.assertEqual(len(metadata["environment"]["uv_lock_sha256"]), 64)
                self.assertIn("revision", metadata["code"])
                outputs = []

                def render(model, rays_o, *args, **kwargs):
                    self.assertEqual(kwargs["device"], torch.device("cpu"))
                    self.assertEqual(kwargs["netchunk"], 2)
                    with torch.no_grad():
                        outputs.append(model(positions, directions))
                    return torch.zeros(len(rays_o), 3)

                argv = [
                    "eval.py",
                    "--checkpoint",
                    str(path),
                    "--output",
                    str(self.root / name),
                    "--device",
                    "cpu",
                    "--no-compile",
                    "--netchunk",
                    "2",
                ]
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    patch("sys.argv", argv),
                    patch.object(torch.cuda, "is_available", return_value=False),
                    patch.object(
                        evaluate,
                        "load_configured_scene",
                        side_effect=AssertionError("must use saved intrinsics"),
                    ),
                    patch("wavelet_nerf.rendering.render_nerf", render),
                ):
                    evaluate.main()
                self.assertEqual(len(outputs), 1)
                expected = original(positions, directions)
                for actual, reference in zip(outputs[0], expected):
                    torch.testing.assert_close(actual, reference)
                self.assertTrue((self.root / name / "frame_0000.png").exists())

    def test_conflicting_settings_fail_and_runtime_overrides_work(self):
        config = resolve_experiment_config(
            {"model_type": "wavelet", "omega0": 11.0, "experiment_name": "test"}
        )
        checkpoint = {"experiment": {"config": config}}
        for key, value in [
            ("omega0", 5),
            ("normalized", False),
            ("near", 1),
            ("model_type", "siren"),
            ("learning_rate", 0.1),
            ("seed", 7),
        ]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                resolve_experiment_config({key: value}, checkpoint, training=True)
        restored = resolve_experiment_config(
            {"omega0": 11.0, "num_iters": 200000, "num_samples_eval": 16},
            checkpoint,
            training=True,
        )
        self.assertEqual(restored["num_samples_eval"], 16)
        self.assertEqual(restored["wave_hidden_dim"], 256)

    def test_changed_dataset_is_rejected(self):
        checkpoint = {"experiment": {"dataset": {"splits": self.splits}}}
        check_dataset(checkpoint, self.splits)
        altered = {
            "train": {**self.splits["train"], "indices": [1]},
            "val": self.splits["val"],
        }
        with self.assertRaisesRegex(ValueError, "dataset"):
            check_dataset(checkpoint, altered)

    def test_cpu_metadata_keeps_build_suffix_and_plain_checkpoint_values(self):
        with patch("torch.cuda.is_available", return_value=False):
            metadata = experiment_metadata(
                {}, self.splits, SceneNormalization(), dataset_metadata(), device="cpu"
            )
        environment = metadata["environment"]
        self.assertEqual(environment["torch_build"], str(torch.__version__))
        self.assertIs(type(environment["torch_build"]), str)
        self.assertEqual(environment["cuda_runtime"], torch.version.cuda)
        self.assertEqual(environment["device"], "cpu")
        self.assertEqual(environment["gpus"], [])
        self.assertIsNone(environment["nvidia_driver"])
        path = self.root / "environment.pth"
        torch.save(environment, path)
        self.assertEqual(torch.load(path, weights_only=True), environment)

    def test_cuda_metadata_records_visible_devices_and_driver(self):
        properties = SimpleNamespace(
            name="Test GPU", total_memory=8192, major=8, minor=6
        )
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=properties),
            patch(
                "wavelet_nerf.experiment.subprocess.check_output",
                return_value="580.00\n",
            ),
        ):
            metadata = accelerator_metadata(torch.device("cuda:0"))
        self.assertEqual(metadata["device"], "cuda:0")
        self.assertEqual(metadata["nvidia_driver"], "580.00")
        self.assertEqual(
            metadata["gpus"],
            [{"name": "Test GPU", "memory_bytes": 8192, "compute_capability": [8, 6]}],
        )

    def test_missing_driver_utility_does_not_prevent_checkpoint_metadata(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=0),
            patch(
                "wavelet_nerf.experiment.subprocess.check_output",
                side_effect=FileNotFoundError,
            ),
        ):
            self.assertIsNone(accelerator_metadata()["nvidia_driver"])


if __name__ == "__main__":
    unittest.main()
