import argparse
import time
import torch
from nerf.data import get_rays
from nerf.models import NeRFModel

def parse_config(config_path: str) -> dict:
    """
    Parse a simple configuration file where each non-empty, non-comment line is of the format:
        key = value
    Returns:
        A dictionary mapping keys to values.
    """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', maxsplit=1)
            config[key.strip()] = value.strip()
    return config

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Train NeRF on a given dataset.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (e.g. config_lego.txt)')
    args = parser.parse_args()

    # Load configuration file.
    config = parse_config(args.config)
    dataset_path = config['datadir']  # Dataset path is read directly from the config file.

    # Start timing for dataset loading
    load_start_time = time.time()

    # Compute rays using the vectorized function.
    rays_o, rays_d, target_pixels = get_rays(dataset_path, 'train')

    # End timing after dataset loading
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    print(f"Dataset loading time (get_rays.py): {load_duration:.2f} seconds")

    # Print ray shapes
    print(f"Ray origins shape: {rays_o.shape}")
    print(f"Ray directions shape: {rays_d.shape}")
    print(f"Target pixels shape: {target_pixels.shape}")

    # -------------------------------
    # Test the Neural Network
    # -------------------------------

    # Instantiate the NeRF model.
    model = NeRFModel(pos_encoding_dim=10, dir_encoding_dim=4, hidden_dim=256)
    model.eval()  # Set model to evaluation mode.

    # Select a subset of rays for testing (e.g. the first 1024 rays, if available).
    num_test_rays = 1024 if rays_o.shape[0] >= 1024 else rays_o.shape[0]
    test_points = rays_o[:num_test_rays]
    test_rays_d = rays_d[:num_test_rays]

    # Run a forward pass through the network.
    with torch.no_grad():
        test_colors, test_density = model(torch.from_numpy(test_points), 
                                          torch.from_numpy(test_rays_d))

    # Print the output shapes.
    print(f"Test output colors shape: {test_colors.shape}")
    print(f"Test output density shape: {test_density.shape}")

if __name__ == '__main__':
    main()
