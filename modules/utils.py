import os
import datetime
import torch
from tqdm import tqdm
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from modules.loss import mse_to_psnr
from modules.configuration import parse_config  # noqa: F401 - retained public import


def format_elapsed_time(start_time: datetime.datetime) -> str:
    """
    Compute the elapsed time since start_time and format it as HH:MM:SS.
    """
    elapsed_time = datetime.datetime.now() - start_time
    total_seconds = int(elapsed_time.total_seconds())
    return '{:02d}:{:02d}:{:02d}'.format(
        total_seconds // 3600,
        (total_seconds % 3600) // 60,
        total_seconds % 60
    )


def get_checkpoint_step(checkpoint):
    """Return completed updates, including checkpoints with legacy loop-index steps."""
    if checkpoint.get('step_semantics') == 'completed_updates':
        return checkpoint['step']
    # LambdaLR advances once per optimizer update. Unlike the old step field,
    # its last_epoch is consistent for periodic, final, and interrupted saves.
    return checkpoint['scheduler_state_dict']['last_epoch']


def load_checkpoint(checkpoint_path):
    """Load on CPU, accepting legacy torch.compile parameter names and metadata."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version', 1) not in (1, 2):
        raise ValueError(f"Unsupported checkpoint format version: {checkpoint['format_version']}")
    if checkpoint.get('format_version') == 2 and 'experiment' not in checkpoint:
        raise ValueError('Checkpoint format 2 requires experiment metadata')
    state_dict = checkpoint['model_state_dict']
    while any(key.startswith('_orig_mod.') for key in state_dict):
        consume_prefix_in_state_dict_if_present(state_dict, '_orig_mod.')
    return checkpoint


def save_checkpoint(step, model, optimizer, scheduler, save_path, model_type, experiment_name,
                    scene_normalization=None, experiment=None):
    """
    Save a training checkpoint with step equal to completed optimizer updates.
    """
    # Compilation wraps the same parameters. Save the underlying module so
    # destination devices do not need to use the same compilation mode.
    while hasattr(model, '_orig_mod'):
        model = model._orig_mod
    checkpoint_dict = {
        'step': step,
        'step_semantics': 'completed_updates',
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    if scene_normalization is not None:
        checkpoint_dict['scene_normalization'] = scene_normalization.to_dict()
    if experiment is not None:
        checkpoint_dict['format_version'] = 2
        checkpoint_dict['experiment'] = experiment
    model_filename = os.path.join(save_path, f"{experiment_name}_{step:06d}.pth")
    torch.save(checkpoint_dict, model_filename)
    return model_filename


def log_training_metrics(step, scheduler, loss, start_time, writer):
    """
    Log training metrics.
    """
    current_lr = scheduler.get_last_lr()[0]
    elapsed_str = format_elapsed_time(start_time)
    log_message = (f"[{elapsed_str}] [Iter {step:07d}] LR: {current_lr:.6f} "
                   f"MSE: {loss.item():.4f} PSNR: {mse_to_psnr(loss.item()):.2f}")
    tqdm.write(log_message)
    writer.add_scalar('loss', loss.item(), step)
    writer.add_scalar('psnr', mse_to_psnr(loss.item()), step)
    writer.add_scalar('learning_rate', current_lr, step)
