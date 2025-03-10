import os
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nerf.data import load_dataset, compute_rays, RayDataset
from nerf.models import NeRFModel, SirenNeRFModel
from nerf.rendering import render_nerf
from nerf.loss import mse_to_psnr


def parse_config(config_path: str) -> dict:
    """
    Parse a configuration file where each non-empty, non-comment line is of the format:
        key = value  # optional inline comment
    Returns a dictionary mapping keys to values.
    """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            line = line.split('#', 1)[0].strip()

            if not line:
                continue

            if '=' in line:
                key, value = line.split('=', maxsplit=1)
                config[key.strip()] = value.strip()
            else:
                print(f"Warning: Invalid line in config file: {line}")

    return config


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


def save_checkpoint(step, model, optimizer, scheduler, save_path, model_type, prefix=""):
    """
    Save the training checkpoint.
    """
    checkpoint_dict = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    model_filename = os.path.join(save_path, f"{model_type}_model_{prefix}{step:07d}.pth")
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train NeRF on a given dataset using volumetric rendering."
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint file to resume training from')
    args = parser.parse_args()
    config = parse_config(args.config)

    # Dataset parameters
    dataset_path = config.get('dataset_path', './datasets/lego')

    # Sampling Parameters
    num_random_rays = int(config.get('num_random_rays', 1024))
    chunk_size = int(config.get('chunk_size', 8192))
    num_samples = int(config.get('num_samples', 256))

    # Training Parameters
    num_iters = int(config.get('num_iters', 150000))
    learning_rate = float(config.get('learning_rate', 5e-4))
    near = float(config.get('near', 2.0))
    far = float(config.get('far', 6.0))

    # Model saving parameters
    save_path = config.get('save_path', './models')
    save_interval = int(config.get('save_interval', 5000))
    os.makedirs(save_path, exist_ok=True)

    # Learning rate decay parameters
    lr_decay = float(config.get('lr_decay', 150))
    lr_decay_factor = float(config.get('lr_decay_factor', 0.1))

    # Model type
    model_type = config.get('model_type', 'NeRF')

    # Monitoring parameters
    log_interval = int(config.get('log_interval', 10))
    val_interval = int(config.get('val_interval', 1000))
    
    print("===== Training Configuration Summary =====")
    print(f"Dataset path: {dataset_path}")
    print(f"Number of random rays: {num_random_rays}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of samples: {num_samples}")
    print(f"Number of iterations: {num_iters}")
    print(f"Learning rate: {learning_rate}")
    print(f"Near plane: {near}")
    print(f"Far plane: {far}")
    print(f"Save path: {save_path}")
    print(f"Save interval: {save_interval}")
    print(f"LR decay: {lr_decay}")
    print(f"LR decay factor: {lr_decay_factor}")
    print(f"Model type: {model_type}")
    print("==========================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == 'nerf':
        model = NeRFModel().to(device)
    elif model_type == 'siren':
        model = SirenNeRFModel().to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Load the full training dataset
    print("Loading training dataset...")
    images_np, c2w_matrices_np, focal_length = load_dataset(dataset_path, mode='train')
    rays_o, rays_d, target_pixels = compute_rays(images_np, c2w_matrices_np, focal_length)

    # Create the dataset and DataLoader
    dataset = RayDataset(rays_o, rays_d, target_pixels)
    data_loader = DataLoader(dataset, batch_size=num_random_rays, shuffle=True)
    loader_iter = iter(data_loader)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # Learning rate scheduler
    gamma = lr_decay_factor ** (1 / (lr_decay * 1000))
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # TensorBoard writer
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f"./logs/{model_type}_{os.path.basename(dataset_path)}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('config', str(config))

    # Resume training from a checkpoint
    start_iter = 0
    start_time = datetime.datetime.now()
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter = checkpoint['step']
        print(f"Resuming training from iteration {start_iter}")

    # Training loop
    try:
        with tqdm(total=num_iters, initial=start_iter, desc="Training", unit="it") as pbar:
            for step in range(start_iter, num_iters):
                try:
                    # Get the next batch from the DataLoader
                    rays_o_batch, rays_d_batch, target_rgb_batch = next(loader_iter)
                except StopIteration:
                    # Reinitialize iterator if the DataLoader is exhausted
                    loader_iter = iter(data_loader)
                    rays_o_batch, rays_d_batch, target_rgb_batch = next(loader_iter)
                
                rays_o_batch = rays_o_batch.to(device)
                rays_d_batch = rays_d_batch.to(device)
                target_rgb_batch = target_rgb_batch.to(device)
                
                # Render using your model
                pred_rgb = render_nerf(
                    model,
                    rays_o_batch,
                    rays_d_batch,
                    near,
                    far,
                    num_samples=num_samples,
                    device=device,
                    white_background=True,
                    chunk_size=chunk_size
                )

                # Compute loss, backpropagate, and update model
                optimizer.zero_grad()
                loss = mse_loss(pred_rgb, target_rgb_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(1)

                # Log metrics and write to TensorBoard at the specified interval
                if step % log_interval == 0:
                    log_training_metrics(step, scheduler, loss, start_time, writer)

                # Save checkpoints at intervals
                if step % save_interval == 0 and step > 0 and step < num_iters - 1:
                    model_filename = save_checkpoint(step, model, optimizer, scheduler, save_path, model_type)
                    elapsed_str = format_elapsed_time(start_time)
                    tqdm.write(f"[{elapsed_str}] Model saved to {model_filename} at iteration {step}")

            # Save final model after training is complete
            final_model_path = os.path.join(save_path, f"{model_type}_model_final.pth")
            final_checkpoint_dict = {
                'step': num_iters,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(final_checkpoint_dict, final_model_path)
            elapsed_str = format_elapsed_time(start_time)
            tqdm.write(f"[{elapsed_str}] Training complete!")
            tqdm.write(f"[{elapsed_str}] Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        # Save checkpoint on keyboard interrupt
        elapsed_str = format_elapsed_time(start_time)
        tqdm.write(f"\n[{elapsed_str}] Keyboard interrupt detected! Saving current checkpoint...")
        interrupt_checkpoint_path = save_checkpoint(step, model, optimizer, scheduler, save_path, model_type, prefix="interrupt_")
        tqdm.write(f"[{elapsed_str}] Checkpoint saved to {interrupt_checkpoint_path}. Exiting training.")


if __name__ == '__main__':
    main()
