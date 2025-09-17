import os
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.data import load_dataset, compute_rays, RayDataset
from modules.models import NeRF, Siren
from modules.models import WaveletNeRF
from modules.rendering import render_nerf
from modules.loss import mse_to_psnr
from modules.utils import parse_config, format_elapsed_time
from modules.utils import save_checkpoint, log_training_metrics


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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint file to resume training from')
    args = parser.parse_args()
    config = parse_config(args.config)

    # Dataset parameters
    dataset_path = config.get('dataset_path', './datasets/lego')

    # Sampling parameters
    num_random_rays = int(config.get('num_random_rays', 1024))
    chunk_size = int(config.get('chunk_size', 8192))
    num_samples = int(config.get('num_samples', 256))
    num_samples_eval = int(config.get('num_samples_eval', 256))

    # Training parameters
    num_iters = int(config.get('num_iters', 150000))
    learning_rate = float(config.get('learning_rate', 5e-4))
    near = float(config.get('near', 2.0))
    far = float(config.get('far', 6.0))

    # Log parameters
    log_root = config.get('log_root', './logs')
    os.makedirs(log_root, exist_ok=True)
    experiment_name = config.get('experiment_name')
    if not experiment_name:
        raise ValueError("Config must define an 'experiment_name' field")
    log_dir = os.path.join(log_root, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Model saving parameters
    save_root = config.get('save_path', './models')
    save_path = os.path.join(save_root, experiment_name)
    save_interval = int(config.get('save_interval', 5000))
    os.makedirs(save_path, exist_ok=True)

    # Learning rate decay parameters
    lr_decay = float(config.get('lr_decay', 150))
    lr_decay_factor = float(config.get('lr_decay_factor', 0.1))
    lr_min = float(config.get('lr_min', 1e-5))

    # First step render flag
    first_step_render = config.get('first_step_render', 'False').lower() == 'true'

    # Resume training from checkpoint if specified
    if args.resume is not None:
        checkpoint_temp = torch.load(args.resume, map_location='cpu', weights_only=True)
        model_type = checkpoint_temp.get('model_type', config.get('model_type', 'NeRF')).lower()
        print(f"Resuming training with model type from checkpoint: {model_type}")
    else:
        existing_logs  = os.listdir(log_dir)
        existing_ckpts = os.listdir(save_path)
        if existing_logs or existing_ckpts:
            print(f"WARNING: Experiment '{experiment_name}' already contains data:")
            if existing_logs:
                print(f"  • {len(existing_logs)} files in log directory: {log_dir}")
            if existing_ckpts:
                print(f"  • {len(existing_ckpts)} files in checkpoint directory: {save_path}")
            print("Starting a new run will append logs and may overwrite old checkpoints.")
            confirm = input("Continue and overwrite existing data? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborting.")
                exit(0)

        model_type = config.get('model_type', 'nerf').lower()

    # Depending on model type, pull out hyperparameters
    if model_type == 'nerf':
        pos_encoding_dim = int(config.get('pos_encoding_dim', 10))
        dir_encoding_dim = int(config.get('dir_encoding_dim', 4))
        hidden_dim     = int(config.get('hidden_dim', 256))
        model = NeRF(
            pos_encoding_dim=pos_encoding_dim,
            dir_encoding_dim=dir_encoding_dim,
            hidden_dim=hidden_dim
        ).to(device)

    elif model_type == 'siren':
        num_layers            = int(config.get('num_layers', 8))
        hidden_dim            = int(config.get('siren_hidden_dim', 256))
        dir_encoding_dim      = int(config.get('siren_dir_encoding_dim', 4))
        sigma_mul             = float(config.get('sigma_mul', 10.0))
        rgb_mul               = float(config.get('rgb_mul', 1.0))
        w0                    = float(config.get('w0', 30.0))
        hidden_w0             = float(config.get('hidden_w0', 1.0))

        model = Siren(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dir_encoding_dim=dir_encoding_dim,
            sigma_mul=sigma_mul,
            rgb_mul=rgb_mul,
            w0=w0,
            hidden_w0=hidden_w0
        ).to(device)

    elif model_type == 'wavelet':
        in_features        = int(config.get('wave_in_features', 3))
        hidden_dim         = int(config.get('wave_hidden_dim', 256))
        num_layers         = int(config.get('wave_num_layers', 8))
        dir_encoding_dim   = int(config.get('wave_dir_encoding_dim', 4))
        input_scale        = float(config.get('input_scale', 256.0))
        weight_scale       = float(config.get('weight_scale', 1.0))
        alpha              = float(config.get('alpha', 6.0))
        beta               = float(config.get('beta', 0.5))
        omega0             = float(config.get('omega0', 5.0))
        normalized_flag    = config.get('normalized', 'True').lower() in ['true', '1', 'yes']

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

    # Monitoring parameters
    log_interval = int(config.get('log_interval', 10))
    val_interval = int(config.get('val_interval', 1000))
    print("\n===== Training Configuration Summary =====")
    print(f"Experiment name: {experiment_name}")
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
    print(f"LR min: {lr_min}")
    print(f"First step render: {first_step_render}")
    print(f"Log interval: {log_interval}")
    print(f"Validation interval: {val_interval}")
    print(f"Log directory: {log_dir}")
    print(f"===========================================")
    print("\n========== Model Hyperparameters ==========")
    print(f"Model type: {model_type}")

    if model_type == 'nerf':
        print(f"pos_encoding_dim: {pos_encoding_dim}")
        print(f"dir_encoding_dim: {dir_encoding_dim}")
        print(f"hidden_dim: {hidden_dim}")

    elif model_type == 'siren':
        print(f"num_layers: {num_layers}")
        print(f"siren_hidden_dim: {hidden_dim}")
        print(f"siren_dir_encoding_dim: {dir_encoding_dim}")
        print(f"sigma_mul: {sigma_mul}")
        print(f"rgb_mul: {rgb_mul}")
        print(f"w0: {w0}")
        print(f"hidden_w0: {hidden_w0}")

    elif model_type == 'wavelet':
        print(f"wave_in_features: {in_features}")
        print(f"wave_hidden_dim: {hidden_dim}")
        print(f"wave_num_layers: {num_layers}")
        print(f"wave_dir_encoding_dim: {dir_encoding_dim}")
        print(f"input_scale: {input_scale}")
        print(f"weight_scale: {weight_scale}")
        print(f"alpha: {alpha}")
        print(f"beta: {beta}")
        print(f"omega0: {omega0}")
        print(f"normalized: {normalized_flag}")

    print(f"===========================================\n")
    
    # Compile the model for performance optimization
    if device.type == 'cuda':
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    else:
        print("Skipping torch.compile: CPU-only environment")

    # Load the training dataset
    print("Loading training dataset...")
    images_np, c2w_matrices_np, focal_length = load_dataset(dataset_path, mode='train')
    rays_o, rays_d, target_pixels = compute_rays(images_np, c2w_matrices_np, focal_length)

    # Load the validation dataset
    print("Loading validation dataset...")
    images_val_np, c2w_val_np, focal_length_val = load_dataset(dataset_path, mode='val')
    N_val, H_val, W_val, _ = images_np.shape

    # Create the dataset and DataLoader
    dataset = RayDataset(rays_o, rays_d, target_pixels)
    data_loader = DataLoader(dataset, 
                             batch_size=num_random_rays, 
                             shuffle=True,
                             num_workers=4,
                             pin_memory=(device.type == 'cuda'))
    loader_iter = iter(data_loader)

    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # Learning rate scheduler
    gamma = lr_decay_factor ** (1 / (lr_decay * 1000))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(gamma**step, lr_min / learning_rate)
    )

    # TensorBoard writer
    writer_kwargs = {'log_dir': log_dir}
    start_iter = 0
    start_time = datetime.datetime.now()
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter = checkpoint['step']
        print(f"Resuming training from iteration {start_iter}")
        writer_kwargs['purge_step'] = start_iter
    writer = SummaryWriter(**writer_kwargs)
    writer.add_text('config', str(config))

    # Training loop
    try:
        with tqdm(total=num_iters, initial=start_iter, desc="Training", unit="it") as pbar:
            for step in range(start_iter, num_iters):
                try:
                    rays_o_batch, rays_d_batch, target_rgb_batch = next(loader_iter)
                except StopIteration:
                    # Reset the iterator if it reaches the end of the dataloader
                    loader_iter = iter(data_loader)
                    rays_o_batch, rays_d_batch, target_rgb_batch = next(loader_iter)
                
                rays_o_batch = rays_o_batch.to(device)
                rays_d_batch = rays_d_batch.to(device)
                target_rgb_batch = target_rgb_batch.to(device)
                
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

                # Log metrics and write to TensorBoard
                if step % log_interval == 0:
                    log_training_metrics(step, scheduler, loss, start_time, writer)

                # Save checkpoint
                if step % save_interval == 0 and step > 0 and step < num_iters - 1:
                    model_filename = save_checkpoint(step, 
                                                     model, 
                                                     optimizer, 
                                                     scheduler, 
                                                     save_path, 
                                                     model_type,
                                                     experiment_name)
                    elapsed_str = format_elapsed_time(start_time)
                    tqdm.write(f"[{elapsed_str}] Model saved to {model_filename} at iteration {step}")

                # Log validation metrics
                if step % val_interval == 0 and (step > 0 or first_step_render):
                    # Select a random image and render it for validation
                    test_image_index = np.random.randint(N_val)
                    single_val_image = images_val_np[test_image_index:test_image_index+1]
                    single_val_c2w = c2w_val_np[test_image_index:test_image_index+1]
                    rays_o_val_np, rays_d_val_np, _ = compute_rays(single_val_image, single_val_c2w, focal_length_val)
                    rays_o_val = torch.from_numpy(rays_o_val_np).float().to(device).squeeze(0)
                    rays_d_val = torch.from_numpy(rays_d_val_np).float().to(device).squeeze(0)

                    tqdm.write("Rendering validation image...")
                    
                    model.eval()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    with torch.no_grad():
                        pred_val_rgb = render_nerf(
                            model,
                            rays_o_val,
                            rays_d_val,
                            near,
                            far,
                            num_samples=num_samples_eval,
                            device=device,
                            white_background=True,
                            chunk_size=chunk_size,
                            stratified=False
                        )
                    model.train()
                    
                    # Reshape to image
                    H_val, W_val = single_val_image.shape[1:3]
                    pred_val_rgb = pred_val_rgb.reshape(H_val, W_val, 3).cpu().numpy()
                    tqdm.write(f"Validation Debug: Rendered image shape: {pred_val_rgb.shape}")
                    
                    # Compute validation PSNR
                    gt_val_img = single_val_image[0]
                    val_mse = np.mean((pred_val_rgb - gt_val_img) ** 2)
                    val_psnr = mse_to_psnr(val_mse)
                    tqdm.write(f"Validation Debug: MSE = {val_mse:.4f}, PSNR = {val_psnr:.2f}")
                    writer.add_scalar("val/psnr", val_psnr, step)
                    
                    # Log the rendered image as a TensorBoard image
                    pred_val_rgb_clamped = np.clip(pred_val_rgb, 0.0, 1.0)
                    writer.add_image(
                        "val/render",
                        torch.from_numpy(pred_val_rgb_clamped).permute(2, 0, 1),
                        step
                    )
                    
                    tqdm.write(f"Validation Debug: Logging complete for iteration {step}.")
                    tqdm.write(f"[Validation Step] Iter {step}  PSNR: {val_psnr:.2f}")

                # Update progress bar
                pbar.update(1)

            # Save final model after training is complete
            final_model_path = save_checkpoint(num_iters, 
                                               model, 
                                               optimizer, 
                                               scheduler, 
                                               save_path, 
                                               model_type,
                                               experiment_name)
            elapsed_str = format_elapsed_time(start_time)
            tqdm.write(f"[{elapsed_str}] Training complete!")
            tqdm.write(f"[{elapsed_str}] Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        # Save checkpoint on keyboard interrupt
        elapsed_str = format_elapsed_time(start_time)
        tqdm.write(f"\n[{elapsed_str}] Keyboard interrupt detected! Saving current checkpoint...")
        interrupt_checkpoint_path = save_checkpoint(step, 
                                                    model, 
                                                    optimizer, 
                                                    scheduler, 
                                                    save_path, 
                                                    model_type,
                                                    experiment_name)
        tqdm.write(f"[{elapsed_str}] Checkpoint saved to {interrupt_checkpoint_path}. Exiting training.")


if __name__ == '__main__':
    main()
