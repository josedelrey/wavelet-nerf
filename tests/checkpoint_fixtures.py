"""Complete current-format checkpoint fixtures for isolated unit tests."""

import numpy as np

from wavelet_nerf.datasets.types import SceneData
from wavelet_nerf.data import PixelRaySampler
from wavelet_nerf.experiment import experiment_metadata, resolve_experiment_config
from wavelet_nerf.run_state import capture_rng
from wavelet_nerf.scene import SceneNormalization
from wavelet_nerf.utils import save_checkpoint


def dataset_metadata():
    return {
        "world_to_scene": np.eye(4).tolist(),
        "render_intrinsics": {
            "height": 2,
            "width": 2,
            "matrix": [[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]],
        },
        "render_path": {"type": "orbit", "radius": 4.0, "elevation": -30.0},
    }


def save_test_checkpoint(
    step,
    model,
    optimizer,
    scheduler,
    save_path,
    model_type,
    name,
    *,
    scene_normalization=None,
    experiment=None,
):
    model_type = "nerf" if model_type == "test" else model_type
    transform = scene_normalization or SceneNormalization()
    config = resolve_experiment_config({"model_type": model_type})
    cameras = np.eye(4, dtype=np.float32)[None]
    scene = SceneData(
        "blender",
        np.zeros((1, 2, 2, 3), dtype=np.float32),
        cameras,
        np.array([[[2.0, 0, 1], [0, 2.0, 1], [0, 0, 1]]], dtype=np.float32),
        np.array([[2.0, 6.0]], dtype=np.float32),
        {key: np.array([0]) for key in ("train", "val", "test")},
        np.array([0]),
        ("view.png",),
        np.eye(4, dtype=np.float32),
        (2.0, 6.0),
        True,
        cameras,
        {"type": "orbit", "radius": 4.0, "elevation": -30.0},
    )
    metadata = experiment_metadata(
        config, scene.describe()["splits"], transform, dataset_metadata=scene.describe()
    )
    if experiment is not None:
        metadata.update(experiment)
        metadata["dataset"] = {**scene.describe(), **experiment["dataset"]}
    return save_checkpoint(
        step,
        model,
        optimizer,
        scheduler,
        save_path,
        model_type,
        name,
        scene_normalization=transform,
        experiment=metadata,
        training_state={
            "rng": capture_rng(),
            "sampler": PixelRaySampler(
                scene.images, scene.poses, scene.intrinsics
            ).state_dict(),
        },
    )
