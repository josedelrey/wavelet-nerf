import numpy as np
import torch
import argparse
import os
import imageio
from tqdm import tqdm

from nerf.data import load_dataset, compute_rays
from nerf.models import NeRF, Siren
from nerf.rendering import render_nerf


def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


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


def main():
    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train NeRF on a given dataset using volumetric rendering."
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='rendered_frames',
                        help='Path to output directory')
    args = parser.parse_args()
    config = parse_config(args.config)

    # Parameters
    dataset_path = config.get('dataset_path', './datasets/lego')
    checkpoint_temp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model_type = checkpoint_temp.get('model_type', config.get('model_type', 'NeRF')).lower()
    model_path = args.checkpoint
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    near = float(config.get('near', 2.0))
    far = float(config.get('far', 6.0))
    num_samples = int(config.get('num_samples', 256))
    chunk_size = int(config.get('chunk_size', 8192))
    num_render_poses = int(config.get('num_render_poses', 40))

    print("===== Evaluation Configuration Summary =====")
    print(f"Dataset path: {dataset_path}")
    print(f"Model type: {model_type}")
    print(f"Model path: {model_path}")
    print(f"Log directory: {output_dir}")
    print(f"Near: {near}")
    print(f"Far: {far}")
    print(f"Num samples: {num_samples}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of render poses: {num_render_poses}")
    print("=============================================")

    # Generate render poses
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, num_render_poses + 1)[:-1]
        ],
        0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    if model_type == 'nerf':
        model = NeRF().to(device)
    elif model_type == 'siren':
        model = Siren().to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    images_val_np, _, focal_length = load_dataset(dataset_path, mode='test', single_image=True)
    single_val_image = images_val_np[0:1]

    # Initialize tqdm for the rendering loop
    render_loop = tqdm(
        range(render_poses.shape[0]),
        desc="Rendering frames",
        unit="frame",
        dynamic_ncols=True  # Adjusts width to terminal
    )

    for i in render_loop:
        single_val_c2w = render_poses[i:i + 1]
        rays_o_val_np, rays_d_val_np, _ = compute_rays(single_val_image, single_val_c2w, focal_length)
        rays_o_val = torch.from_numpy(rays_o_val_np).float().to(device).squeeze(0)
        rays_d_val = torch.from_numpy(rays_d_val_np).float().to(device).squeeze(0)
        
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_val_rgb = render_nerf(
                model,
                rays_o_val,
                rays_d_val,
                near,
                far,
                num_samples=num_samples,
                device=device,
                white_background=True,
                chunk_size=chunk_size
            )
        model.train()
        
        # Reshape to image
        H_val, W_val = single_val_image.shape[1:3]
        pred_val_rgb = pred_val_rgb.reshape(H_val, W_val, 3).cpu().numpy()
        
        # Log the rendered image as a TensorBoard image
        pred_val_rgb_clamped = np.clip(pred_val_rgb, 0.0, 1.0)
        frame = (pred_val_rgb_clamped * 255).astype(np.uint8)

        # Save frame as PNG
        frame_filename = os.path.join(output_dir, f"frame_{i:04d}.png")
        imageio.imwrite(frame_filename, frame)


if __name__ == '__main__':
    main()
