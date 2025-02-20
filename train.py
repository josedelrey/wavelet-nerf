import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from nerf.data import load_dataset, compute_rays
from nerf.models import NeRFModel
from nerf.rendering import render_volume
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
                continue  # Skip empty lines and full-line comments

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
    num_random_rays = int(config.get('num_random_rays', 1024))
    num_iters = int(config.get('num_iters', 1000000))
    learning_rate = float(config.get('learning_rate', 5e-4))
    near = float(config.get('near', 2.0))
    far = float(config.get('far', 6.0))
    save_path = config.get('save_path', './models')
    save_interval = int(config.get('save_interval', 5000))
    os.makedirs(save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the full training dataset
    images_np, c2w_matrices_np, focal_length = load_dataset(dataset_path, mode='train')
    N = images_np.shape[0]

    # Initialize NeRF model and optimizer
    model = NeRFModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # Learning rate scheduler with exponential decay
    lr_decay = float(config.get('lr_decay', 250))
    lr_decay_factor = float(config.get('lr_decay_factor', 0.1))
    gamma = lr_decay_factor ** (1 / (lr_decay * 1000))
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # Training loop
    for step in range(num_iters):
        # Randomly select an image from the dataset
        img_idx = np.random.randint(0, N)

        # Compute rays for the selected image
        with torch.no_grad():
            single_image = images_np[img_idx:img_idx+1]
            single_c2w = c2w_matrices_np[img_idx:img_idx+1]
            rays_o_np, rays_d_np, target_pixels_np = compute_rays(
                single_image, single_c2w, focal_length
            )
        rays_o = torch.from_numpy(rays_o_np).float().to(device).squeeze(0)
        rays_d = torch.from_numpy(rays_d_np).float().to(device).squeeze(0)
        target_pixels = torch.from_numpy(target_pixels_np).float().to(device).squeeze(0)

        # Randomly sample a subset of rays for this iteration
        num_pixels = rays_o.shape[0]
        sel_inds = np.random.choice(num_pixels, size=num_random_rays, replace=False)
        rays_o_batch = rays_o[sel_inds]
        rays_d_batch = rays_d[sel_inds]
        target_rgb = target_pixels[sel_inds]

        # Use render_volume to compute the predicted color along each ray
        pred_rgb = render_volume(
            model,
            rays_o_batch,
            rays_d_batch,
            near,
            far,
            num_bins=100,
            device=device,
            white_background=True
        )

        # Compute loss and update the model
        optimizer.zero_grad()
        loss = mse_loss(pred_rgb, target_rgb)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log progress
        if step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Iter {step:06d}] LR: {current_lr:.6f} "
                  f"MSE: {loss.item():.4f} PSNR: {mse_to_psnr(loss.item()):.2f}")

        # Save model checkpoint
        if step % save_interval == 0 and step > 0:
            model_filename = os.path.join(save_path, f"nerf_model_{step:06d}.pth")
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved to {model_filename} at iteration {step}")

    # Save final model
    final_model_path = os.path.join(save_path, "nerf_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete!")
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
