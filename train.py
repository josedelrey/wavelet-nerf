import argparse
from nerf.data import get_rays

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

    # Compute rays using the vectorized function.
    rays_o, rays_d, target_pixels = get_rays(dataset_path, 'train')

    print(f"Ray origins shape: {rays_o.shape}")
    print(f"Ray directions shape: {rays_d.shape}")
    print(f"Target pixels shape: {target_pixels.shape}")

if __name__ == '__main__':
    main()
