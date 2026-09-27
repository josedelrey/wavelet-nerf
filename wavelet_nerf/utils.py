import os
import datetime
import torch
from tqdm import tqdm

from wavelet_nerf.run_state import atomic_write
from wavelet_nerf.loss import mse_to_psnr
from wavelet_nerf.experiment import resolve_experiment_config


def format_elapsed_time(start_time: datetime.datetime) -> str:
    """
    Compute the elapsed time since start_time and format it as HH:MM:SS.
    """
    elapsed_time = datetime.datetime.now() - start_time
    total_seconds = int(elapsed_time.total_seconds())
    return "{:02d}:{:02d}:{:02d}".format(
        total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
    )


CHECKPOINT_FORMAT_VERSION = 3
_CHECKPOINT_FIELDS = {
    "format_version",
    "step",
    "model_type",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scene_normalization",
    "experiment",
    "training_state",
}


def _validate_checkpoint(checkpoint):
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            f"Checkpoint must use format version {CHECKPOINT_FORMAT_VERSION}"
        )
    missing = _CHECKPOINT_FIELDS - checkpoint.keys()
    unknown = checkpoint.keys() - _CHECKPOINT_FIELDS
    if missing or unknown:
        raise ValueError(
            f"Invalid checkpoint fields: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    if type(checkpoint["step"]) is not int or checkpoint["step"] < 0:
        raise ValueError(
            "Checkpoint step must be a nonnegative count of completed updates"
        )
    for name in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scene_normalization",
        "experiment",
        "training_state",
    ):
        if not isinstance(checkpoint[name], dict):
            raise ValueError(f"Checkpoint {name} must be a mapping")
    experiment = checkpoint["experiment"]
    if not isinstance(experiment.get("config"), dict) or not isinstance(
        experiment.get("dataset"), dict
    ):
        raise ValueError("Checkpoint requires experiment config and dataset metadata")
    dataset = experiment["dataset"]
    if (
        not {"splits", "render_intrinsics", "render_path", "world_to_scene"}
        <= dataset.keys()
    ):
        raise ValueError(
            "Checkpoint requires explicit dataset cameras, preprocessing and render path"
        )
    if not {"rng", "sampler"} <= checkpoint["training_state"].keys():
        raise ValueError("Checkpoint requires training RNG and sampler state")
    rng = checkpoint["training_state"]["rng"]
    sampler = checkpoint["training_state"]["sampler"]
    if (
        not isinstance(rng, dict)
        or not {"python", "numpy", "torch", "cuda"} <= rng.keys()
    ):
        raise ValueError(
            "Checkpoint requires complete Python, NumPy, Torch and CUDA RNG state"
        )
    if (
        not isinstance(sampler, dict)
        or sampler.get("protocol") != "independent_pixel_batches_v1"
        or not {"height", "width", "camera_indices", "rng"} <= sampler.keys()
    ):
        raise ValueError("Checkpoint requires current pixel sampler state")
    if not {"center", "scale"} <= checkpoint["scene_normalization"].keys():
        raise ValueError("Checkpoint requires explicit scene normalization")
    if any(key.startswith("_orig_mod.") for key in checkpoint["model_state_dict"]):
        raise ValueError("Checkpoint weights must use canonical unwrapped model names")
    config = resolve_experiment_config({}, checkpoint)
    if checkpoint["model_type"] != config["model_type"]:
        raise ValueError(
            "Checkpoint model_type must match the resolved experiment config"
        )


def load_checkpoint(checkpoint_path):
    """Load and validate the current checkpoint format on CPU."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint)
    return checkpoint


def save_checkpoint(
    step,
    model,
    optimizer,
    scheduler,
    save_path,
    model_type,
    experiment_name,
    *,
    scene_normalization,
    experiment,
    training_state,
):
    """
    Save a training checkpoint with step equal to completed optimizer updates.
    """
    # Compilation wraps the same parameters. Save the underlying module so
    # destination devices do not need to use the same compilation mode.
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    checkpoint_dict = {
        "step": step,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    checkpoint_dict.update(
        scene_normalization=scene_normalization.to_dict(),
        experiment=experiment,
        training_state=training_state,
    )
    _validate_checkpoint(checkpoint_dict)
    model_filename = os.path.join(save_path, f"{experiment_name}_{step:06d}.pth")
    atomic_write(model_filename, lambda file: torch.save(checkpoint_dict, file))
    return model_filename


def log_training_metrics(step, scheduler, loss, start_time, writer):
    """
    Log training metrics.
    """
    current_lr = scheduler.get_last_lr()[0]
    elapsed_str = format_elapsed_time(start_time)
    log_message = (
        f"[{elapsed_str}] [Iter {step:07d}] LR: {current_lr:.6f} "
        f"MSE: {loss.item():.4f} PSNR: {mse_to_psnr(loss.item()):.2f}"
    )
    tqdm.write(log_message)
    writer.add_scalar("loss", loss.item(), step)
    writer.add_scalar("psnr", mse_to_psnr(loss.item()), step)
    writer.add_scalar("learning_rate", current_lr, step)
