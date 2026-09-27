from copy import deepcopy
import json
import socket
import tracemalloc
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from wavelet_nerf.data import (
    CameraRayGenerator,
    PixelRaySampler,
    PixelBatchPlans,
    RayBatchDataset,
    camera_rays,
)
from wavelet_nerf.ndc import ndc_camera_rays
from wavelet_nerf.models import NeRF, Siren, WaveletNeRF
from wavelet_nerf.rendering import render_camera, render_nerf


class MemoryPathTests(unittest.TestCase):
    def cameras(self):
        images = np.arange(3 * 4 * 6 * 3, dtype=np.float32).reshape(3, 4, 6, 3) / 216
        poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        poses[:, 0, 3] = [-0.5, 0, 0.5]
        matrices = np.array(
            [[[4 + i, 0, 3], [0, 5 + i, 2], [0, 0, 1]] for i in range(3)],
            dtype=np.float32,
        )
        return images, poses, matrices

    def test_selected_pixels_match_full_world_and_ndc_rays(self):
        images, poses, matrices = self.cameras()
        cameras, pixels = np.array([2, 0, 1, 2]), np.array([0, 23, 9, 17])
        for kind, normalize in [("blender", True), ("blender", False), ("llff", False)]:
            with self.subTest(kind=kind, normalize=normalize):
                generator = CameraRayGenerator(
                    4, 6, poses, matrices, dataset_type=kind, normalize=normalize
                )
                actual = generator.rays(cameras, pixels)
                full = (
                    ndc_camera_rays(4, 6, poses, matrices)
                    if kind == "llff"
                    else camera_rays(4, 6, poses, matrices, normalize=normalize)
                )
                for batch, expected in zip(actual, full):
                    np.testing.assert_allclose(
                        batch, expected[cameras, pixels], atol=1e-6
                    )
                sampler = PixelRaySampler(
                    images, poses, matrices, dataset_type=kind, normalize=normalize
                )
                batch = sampler.batch(cameras, pixels)
                np.testing.assert_equal(
                    batch[2], images[cameras, pixels // 6, pixels % 6]
                )

    def test_global_sampling_is_reproducible_and_respects_training_split(self):
        images, poses, matrices = self.cameras()
        samplers = [
            PixelRaySampler(images, poses, matrices, image_indices=[0, 2], seed=19)
            for _ in range(2)
        ]
        plans = [sampler.plan(12) for sampler in samplers]
        np.testing.assert_equal(plans[0]["pixels"], plans[1]["pixels"])
        np.testing.assert_equal(plans[0]["cameras"], plans[1]["cameras"])
        self.assertEqual(set(plans[0]["cameras"]), {0, 2})
        self.assertEqual(len(set(zip(plans[0]["cameras"], plans[0]["pixels"]))), 12)

    def test_per_image_crop_matches_reference_center_selection(self):
        images, poses, matrices = self.cameras()
        sampler = PixelRaySampler(images, poses, matrices, seed=9)
        plan = sampler.plan(4, per_image=True, crop_fraction=0.5)
        self.assertEqual(len(set(plan["cameras"])), 1)
        self.assertEqual(set(plan["pixels"]), {8, 9, 14, 15})
        with self.assertRaisesRegex(ValueError, "available"):
            sampler.plan(5, per_image=True, crop_fraction=0.5)

    def test_large_scene_sampling_allocates_batch_sized_geometry(self):
        # Virtual RGB storage represents 64 million pixels without allocating it.
        images = np.broadcast_to(
            np.zeros((1, 1, 1, 3), dtype=np.float32), (100, 800, 800, 3)
        )
        poses = np.tile(np.eye(4, dtype=np.float32), (100, 1, 1))
        matrices = np.array([[600, 0, 400], [0, 600, 400], [0, 0, 1]], dtype=np.float32)
        sampler = PixelRaySampler(images, poses, matrices)
        self.assertIs(sampler.images, images)
        tracemalloc.start()
        try:
            rays = sampler.sample(1024)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual([array.shape for array in rays], [(1024, 3)] * 3)
        self.assertLess(peak, 2 * 1024 * 1024)
        self.assertLess(len(json.dumps(sampler.state_dict())), 4096)

    def test_prefetch_checkpoint_replays_uncommitted_batch(self):
        images, poses, matrices = self.cameras()
        sampler = PixelRaySampler(images, poses, matrices, seed=12)
        before = deepcopy(sampler.state_dict())
        plans = iter(PixelBatchPlans(sampler, 5, 0, 8))
        first, second = next(plans), next(plans)
        # Prefetch advances the live generator, not the saved progress.
        next(plans)
        self.assertEqual(sampler.state_dict(), before)
        sampler.commit(first["rng"])
        restored = PixelRaySampler(images, poses, matrices)
        restored.load_state_dict(sampler.state_dict())
        replay = restored.plan(5)
        np.testing.assert_equal(replay["pixels"], second["pixels"])
        np.testing.assert_equal(replay["cameras"], second["cameras"])
        self.assertEqual(replay["rng"], second["rng"])

    def test_workers_receive_only_batch_plans_and_match_serial_results(self):
        images, poses, matrices = self.cameras()
        results = []
        for workers in (0, 2):
            sampler = PixelRaySampler(images, poses, matrices, seed=17)
            plans = PixelBatchPlans(sampler, 5, 0, 5)
            try:
                if workers:
                    # PyTorch's tensor sharing requires a local Unix socket.
                    with socket.socket(socket.AF_UNIX) as probe:
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                loader = DataLoader(
                    RayBatchDataset(sampler),
                    sampler=plans,
                    batch_size=None,
                    num_workers=workers,
                    generator=torch.Generator().manual_seed(17),
                    timeout=15 if workers else 0,
                    multiprocessing_context="spawn" if workers else None,
                )
                results.append(list(loader))
            except PermissionError as error:
                self.skipTest(f"Multiprocessing unavailable in this sandbox: {error}")
        for serial, parallel in zip(*results):
            self.assertEqual(serial["rng"], parallel["rng"])
            for expected, actual in zip(serial["rays"], parallel["rays"]):
                torch.testing.assert_close(actual, expected)

    def test_cpu_output_buffers_and_camera_chunks_match_normal_rendering(self):
        _, poses, matrices = self.cameras()
        for constructor in (NeRF, Siren, WaveletNeRF):
            for kind in ("blender", "llff"):
                with self.subTest(model=constructor.__name__, dataset=kind):
                    torch.manual_seed(8)
                    model = constructor(hidden_dim=8).eval()
                    generator = CameraRayGenerator(
                        4,
                        6,
                        poses[:1],
                        matrices[:1],
                        dataset_type=kind,
                        normalize=constructor is not NeRF,
                    )
                    rays = generator.rays(np.zeros(24, dtype=np.int64), np.arange(24))
                    options = (
                        {"view_directions": torch.from_numpy(rays[2])}
                        if len(rays) == 3
                        else {}
                    )
                    bounds = (0, 1) if kind == "llff" else (2, 6)
                    with torch.no_grad():
                        expected = render_nerf(
                            model,
                            *map(torch.from_numpy, rays[:2]),
                            *bounds,
                            num_samples=4,
                            chunk_size=5,
                            stratified=False,
                            **options,
                        )
                        buffered = render_nerf(
                            model,
                            *map(torch.from_numpy, rays[:2]),
                            *bounds,
                            num_samples=4,
                            chunk_size=5,
                            stratified=False,
                            output_device="cpu",
                            **options,
                        )
                        camera = render_camera(
                            model,
                            4,
                            6,
                            poses[0],
                            matrices[0],
                            *bounds,
                            num_samples=4,
                            chunk_size=5,
                            dataset_type=kind,
                        )
                    self.assertEqual(buffered.device.type, "cpu")
                    self.assertFalse(buffered.requires_grad)
                    torch.testing.assert_close(buffered, expected)
                    torch.testing.assert_close(camera, expected)

    def test_output_device_cannot_silently_disable_training_gradients(self):
        model = Siren(hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "inference"):
            render_nerf(
                model,
                torch.zeros(2, 3),
                torch.tensor([[0.0, 0.0, -1.0]]).expand(2, -1),
                2,
                6,
                num_samples=4,
                output_device="cpu",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_inference_gpu_memory_does_not_scale_with_total_ray_count(self):
        for constructor in (NeRF, Siren, WaveletNeRF):
            with self.subTest(model=constructor.__name__):
                model = constructor(hidden_dim=32).cuda().eval()
                peaks = []
                for count in (128, 8192):
                    origins = torch.zeros(count, 3)
                    directions = torch.tensor([[0.0, 0.0, -1.0]]).expand(count, -1)
                    options = dict(
                        device="cuda",
                        num_samples=8,
                        num_importance=8,
                        chunk_size=64,
                        netchunk=512,
                        stratified=False,
                        output_device="cpu",
                    )
                    with torch.no_grad():
                        render_nerf(
                            model, origins[:64], directions[:64], 2, 6, **options
                        )
                        torch.cuda.synchronize()
                        baseline = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        rgb = render_nerf(model, origins, directions, 2, 6, **options)
                        torch.cuda.synchronize()
                        peaks.append(torch.cuda.max_memory_allocated() - baseline)
                    self.assertEqual(rgb.device.type, "cpu")
                    self.assertTrue(torch.isfinite(rgb).all())
                self.assertLessEqual(peaks[1], peaks[0] + 65536)


if __name__ == "__main__":
    unittest.main()
