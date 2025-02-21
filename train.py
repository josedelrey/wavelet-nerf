import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from nerf.data import load_dataset, compute_rays
from nerf.models import NeRFModel
from nerf.rendering import render_nerf
from nerf.loss import mse_to_psnr
import datetime
from tqdm import tqdm


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

            # Remove inline comments
            line = line.split('#', 1)[0].strip()

            # Skip lines that become empty after removing comments
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


def main():
    # Load configuration
    parser = argparse.ArgumentParser(
        description="Train NeRF on a given dataset using volumetric rendering."
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (e.g. config_lego.txt)')
    args = parser.parse_args()
    config = parse_config(args.config)

    dataset_path = config.get('datadir', './datasets/lego')
    num_random_rays = int(config.get('num_random_rays', 4096))
    render_batch_size = int(config.get('render_batch_size', 1024))
    num_iters = int(config.get('num_iters', 150000))
    learning_rate = float(config.get('learning_rate', 5e-4))
    near = float(config.get('near', 2.0))
    far = float(config.get('far', 6.0))
    save_path = config.get('save_path', './models')
    save_interval = int(config.get('save_interval', 5000))
    os.makedirs(save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the full training dataset
    images_np, c2w_matrices_np, focal_length = load_dataset(dataset_path, mode='train')
    rays_o, rays_d, target_pixels = compute_rays(images_np, c2w_matrices_np, focal_length)
    N = images_np.shape[0]

    model = NeRFModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # Learning rate scheduler with exponential decay
    lr_decay = float(config.get('lr_decay', 150))
    lr_decay_factor = float(config.get('lr_decay_factor', 0.1))
    gamma = lr_decay_factor ** (1 / (lr_decay * 1000))
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    start_time = datetime.datetime.now()

    # Training loop with tqdm progress bar
    for step in tqdm(range(num_iters), desc="Training", unit="it"):
        # Randomly select an image from the dataset
        img_idx = np.random.randint(0, N)

        # Get the rays and target pixels for the selected image
        rays_o_image_np = rays_o[img_idx]
        rays_d_image_np = rays_d[img_idx]
        target_pixels_image_np = target_pixels[img_idx]
        rays_o_image = torch.from_numpy(rays_o_image_np).float().to(device).squeeze(0)
        rays_d_image = torch.from_numpy(rays_d_image_np).float().to(device).squeeze(0)
        target_pixels_image = torch.from_numpy(target_pixels_image_np).float().to(device).squeeze(0)

        # Randomly sample a subset of rays for this iteration
        num_pixels = rays_o_image.shape[0]
        sel_inds = np.random.choice(num_pixels, size=num_random_rays, replace=False)
        rays_o_batch = rays_o_image[sel_inds]
        rays_d_batch = rays_d_image[sel_inds]
        target_rgb = target_pixels_image[sel_inds]

        # Use render_volume to compute the predicted color along each ray
        pred_rgb = render_nerf(
            model,
            rays_o_batch,
            rays_d_batch,
            near,
            far,
            num_coarse=100,
            device=device,
            white_background=True,
            chunk_size=render_batch_size
        )

        # Compute loss and update the model
        optimizer.zero_grad()
        loss = mse_loss(pred_rgb, target_rgb)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log progress every 10 iterations using elapsed time since training started
        if step % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            elapsed_str = format_elapsed_time(start_time)
            log_message = (f"[{elapsed_str}] [Iter {step:07d}] LR: {current_lr:.6f} "
                           f"MSE: {loss.item():.4f} PSNR: {mse_to_psnr(loss.item()):.2f}")
            tqdm.write(log_message)

        # Save model checkpoint
        if step % save_interval == 0 and step > 0:
            model_filename = os.path.join(save_path, f"nerf_model_{step:07d}.pth")
            torch.save(model.state_dict(), model_filename)
            elapsed_str = format_elapsed_time(start_time)
            tqdm.write(f"[{elapsed_str}] Model saved to {model_filename} at iteration {step}")

    # Save final model
    final_model_path = os.path.join(save_path, "nerf_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    elapsed_str = format_elapsed_time(start_time)
    tqdm.write(f"[{elapsed_str}] Training complete!")
    tqdm.write(f"[{elapsed_str}] Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
