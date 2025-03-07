import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from nerf.data import load_dataset, compute_rays
from nerf.models import NeRFModel
from nerf.rendering import render_nerf


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


def main():
    # Load configuration
    parser = argparse.ArgumentParser(
        description="Evaluate NeRF: Render a test image using a saved checkpoint."
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (e.g. config_lego.txt)')
    args = parser.parse_args()
    config = parse_config(args.config)

    # Load all necessary parameters from the config file
    dataset_path = config.get('datadir', './datasets/lego')
    checkpoint_path = config.get('checkpoint_path', None)
    if checkpoint_path is None:
        raise ValueError("Please provide a checkpoint_path in the config file.")
    near = float(config.get('near', 2.0))
    far = float(config.get('far', 6.0))
    num_samples = int(config.get('num_samples', 256))
    chunk_size = int(config.get('chunk_size', 16384))
    white_background = config.get('white_background', 'True').lower() in ['true', '1', 'yes']
    test_image_index = int(config.get('test_image_index', 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # Load dataset in test mode
    images_np, c2w_matrices_np, focal_length = load_dataset(dataset_path, mode='test')
    N, H, W, _ = images_np.shape
    print(f"Loaded {N} test images of resolution {H}x{W}.")

    # Select the test image using the specified index and compute rays
    single_image = images_np[test_image_index:test_image_index+1]
    single_c2w = c2w_matrices_np[test_image_index:test_image_index+1]
    rays_o_np, rays_d_np, _ = compute_rays(single_image, single_c2w, focal_length)
    rays_o = torch.from_numpy(rays_o_np).float().to(device).squeeze(0)
    rays_d = torch.from_numpy(rays_d_np).float().to(device).squeeze(0)

    # Load the saved model checkpoint
    model = NeRFModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # Render the test image
    with torch.no_grad():
        pred_rgb = render_nerf(
            model,
            rays_o,
            rays_d,
            near,
            far,
            num_samples=num_samples,
            device=device,
            white_background=white_background,
            chunk_size=chunk_size
        )

    # Reshape the predicted rays into an image
    rendered_image = pred_rgb.cpu().numpy().reshape(H, W, 3)

    # Compute PSNR between the rendered image and the ground truth test image
    gt_image = single_image[0]
    mse = np.mean((rendered_image - gt_image) ** 2)
    psnr = -10.0 * np.log10(mse)
    print("PSNR: {:.2f}".format(psnr))

    # Plot the rendered image
    plt.figure(figsize=(8, 8))
    plt.imshow(rendered_image)
    plt.axis('off')
    plt.title('Rendered Test Image')
    plt.show()


if __name__ == '__main__':
    main()
