import numpy as np
import torch
import argparse
import os
import imageio
from tqdm import tqdm

from modules.data import load_dataset, compute_rays
from modules.models import NeRF, Siren, MultiScaleWaveletNeRF
from modules.rendering import render_nerf
from modules.utils import parse_config
from modules.camera import pose_spherical


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
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
    num_samples = int(config.get('num_samples_eval', 256))
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

    # Load the model
    if model_type == 'nerf':
        model = NeRF().to(device)
    elif model_type == 'siren':
        model = Siren().to(device)
    elif model_type == 'multiscalewavelet':
        model = MultiScaleWaveletNeRF().to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load a dummy image to get height and width
    images_val_np, _, focal_length = load_dataset(dataset_path, mode='test', single_image=True)
    single_val_image = images_val_np[0:1]

    # Initialize tqdm for the rendering loop
    render_loop = tqdm(
        range(render_poses.shape[0]),
        desc="Rendering frames",
        unit="frame",
        dynamic_ncols=True
    )
        
    # Render the images
    model.eval()
    for i in render_loop:
        single_val_c2w = render_poses[i:i + 1]
        rays_o_val_np, rays_d_val_np, _ = compute_rays(single_val_image, single_val_c2w, focal_length)
        rays_o_val = torch.from_numpy(rays_o_val_np).float().to(device).squeeze(0)
        rays_d_val = torch.from_numpy(rays_d_val_np).float().to(device).squeeze(0)

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
                chunk_size=chunk_size,
                stratified=False
            )
        
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
