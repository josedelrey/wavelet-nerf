import numpy as np
import torch
import argparse
import os
import imageio
from tqdm import tqdm
from collections import OrderedDict

from modules.data import load_dataset, compute_rays
from modules.models import NeRF, Siren, WaveletNeRF
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

    # ===== Load the model with hyperparameters from config =====
    if model_type == 'nerf':
        # NeRF-specific hyperparameters
        pos_encoding_dim = int(config.get('pos_encoding_dim', 10))
        dir_encoding_dim = int(config.get('dir_encoding_dim', 4))
        hidden_dim       = int(config.get('hidden_dim', 256))
        model = NeRF(
            pos_encoding_dim=pos_encoding_dim,
            dir_encoding_dim=dir_encoding_dim,
            hidden_dim=hidden_dim
        ).to(device)

    elif model_type == 'siren':
        # Siren-specific hyperparameters
        num_layers       = int(config.get('num_layers', 8))
        hidden_dim       = int(config.get('siren_hidden_dim', 256))
        dir_encoding_dim = int(config.get('siren_dir_encoding_dim', 4))
        sigma_mul        = float(config.get('sigma_mul', 10.0))
        rgb_mul          = float(config.get('rgb_mul', 1.0))
        w0               = float(config.get('w0', 30.0))
        hidden_w0        = float(config.get('hidden_w0', 1.0))
        model = Siren(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dir_encoding_dim=dir_encoding_dim,
            sigma_mul=sigma_mul,
            rgb_mul=rgb_mul,
            w0=w0,
            hidden_w0=hidden_w0
        ).to(device)

    elif model_type in ('multiscalewavelet', 'wavelet'):
        # WaveletNeRF-specific hyperparameters
        in_features      = int(config.get('wave_in_features', 3))
        hidden_dim       = int(config.get('wave_hidden_dim', 256))
        num_layers       = int(config.get('wave_num_layers', 8))
        dir_encoding_dim = int(config.get('wave_dir_encoding_dim', 4))
        input_scale      = float(config.get('input_scale', 256.0))
        weight_scale     = float(config.get('weight_scale', 1.0))
        alpha            = float(config.get('alpha', 6.0))
        beta             = float(config.get('beta', 0.5))
        omega0           = float(config.get('omega0', 5.0))
        normalized_flag  = config.get('normalized', 'True').lower() in ['true', '1', 'yes']
        model = WaveletNeRF(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dir_encoding_dim=dir_encoding_dim,
            input_scale=input_scale,
            weight_scale=weight_scale,
            alpha=alpha,
            beta=beta,
            omega0=omega0,
            normalized=normalized_flag
        ).to(device)

    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Load the model checkpoint
    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    raw_state = ckpt['model_state_dict']
    clean_state = OrderedDict()
    for k, v in raw_state.items():
        new_k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        clean_state[new_k] = v

    # Load into your model
    model.load_state_dict(clean_state)

    # Now compile for CUDA (if available)
    if device.type == 'cuda':
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    else:
        print("Skipping torch.compile: CPU-only environment")
    
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
