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

from modules.data import load_configured_scene, resolve_sampling_bounds, compute_rays, RayDataset
from modules.ndc import compute_ndc_rays
from modules.models import NeRF, LegacyNeRF, Siren
from modules.models import WaveletNeRF
from modules.rendering import render_nerf
from modules.scene import SceneNormalization, resolve_scene_normalization
from modules.nerf_reference import KerasAdam
from modules.experiment import (resolve_experiment_config,
                                check_dataset, experiment_metadata)
from modules.loss import mse_to_psnr
from modules.utils import parse_config, format_elapsed_time
from modules.utils import load_checkpoint, save_checkpoint, log_training_metrics, get_checkpoint_step


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train NeRF on a given dataset using volumetric rendering."
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint file to resume training from')
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = resolve_experiment_config(parse_config(args.config), checkpoint, training=True)

    # Reproducibility
    seed = int(config['seed'])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Dataset parameters
    dataset_path = config['dataset_path']
    print('Loading scene...')
    scene = load_configured_scene(config, splits=('train', 'val'))
    training, validation = scene.split('train'), scene.split('val')
    ray_function = compute_ndc_rays if scene.dataset_type == 'llff' else compute_rays

    # Sampling parameters
    num_random_rays = int(config['num_random_rays'])
    chunk_size = int(config['chunk_size'])
    num_samples = int(config['num_samples'])
    num_samples_eval = int(config['num_samples_eval'])

    # Training parameters
    num_iters = int(config['num_iters'])
    learning_rate = float(config['learning_rate'])
    near, far = resolve_sampling_bounds(config, scene)

    # Log parameters
    log_root = config['log_root']
    os.makedirs(log_root, exist_ok=True)
    experiment_name = config.get('experiment_name')
    if not experiment_name:
        raise ValueError("Config must define an 'experiment_name' field")
    log_dir = os.path.join(log_root, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Model saving parameters
    save_root = config['save_path']
    save_path = os.path.join(save_root, experiment_name)
    save_interval = int(config['save_interval'])
    os.makedirs(save_path, exist_ok=True)

    # Learning rate decay parameters
    lr_decay = float(config['lr_decay'])
    lr_decay_factor = float(config['lr_decay_factor'])
    lr_min = float(config['lr_min'])

    # First step render flag
    first_step_render = config['first_step_render'].lower() == 'true'

    # Resume training from checkpoint if specified
    if args.resume is not None:
        model_type = config['model_type']
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

    scene_normalization = resolve_scene_normalization(config, checkpoint)
    reference_baseline = model_type == 'nerf' and config['baseline_version'] == 'reference'
    if reference_baseline:
        scene_normalization = SceneNormalization()
    config['scene_center'] = ', '.join(map(str, scene_normalization.center))
    config['scene_scale'] = str(scene_normalization.scale)

    # Depending on model type, pull out hyperparameters
    if model_type == 'nerf':
        pos_encoding_dim = int(config['pos_encoding_dim'])
        dir_encoding_dim = int(config['dir_encoding_dim'])
        hidden_dim     = int(config['hidden_dim'])
        constructor = NeRF if reference_baseline else LegacyNeRF
        options = {'num_importance': int(config['num_importance'])} if reference_baseline else {}
        model = constructor(
            pos_encoding_dim=pos_encoding_dim,
            dir_encoding_dim=dir_encoding_dim,
            hidden_dim=hidden_dim, **options
        ).to(device)

    elif model_type == 'siren':
        num_layers            = int(config['num_layers'])
        hidden_dim            = int(config['siren_hidden_dim'])
        dir_encoding_dim      = int(config['siren_dir_encoding_dim'])
        sigma_mul             = float(config['sigma_mul'])
        rgb_mul               = float(config['rgb_mul'])
        w0                    = float(config['w0'])
        hidden_w0             = float(config['hidden_w0'])

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
        in_features        = int(config['wave_in_features'])
        hidden_dim         = int(config['wave_hidden_dim'])
        num_layers         = int(config['wave_num_layers'])
        dir_encoding_dim   = int(config['wave_dir_encoding_dim'])
        input_scale        = float(config['input_scale'])
        weight_scale       = float(config['weight_scale'])
        alpha              = float(config['alpha'])
        beta               = float(config['beta'])
        omega0             = float(config['omega0'])
        normalized_flag    = config['normalized'].lower() in ['true', '1', 'yes']

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
    log_interval = int(config['log_interval'])
    val_interval = int(config['val_interval'])
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
    print(f"Scene center: {scene_normalization.center}")
    print(f"Scene scale: {scene_normalization.scale}")
    print(f"Save path: {save_path}")
    print(f"Save interval: {save_interval}")
    print(f"LR decay: {lr_decay}")
    print(f"LR decay factor: {lr_decay_factor}")
    print(f"LR min: {lr_min}")
    print(f"First step render: {first_step_render}")
    print(f"Log interval: {log_interval}")
    print(f"Validation interval: {val_interval}")
    print(f"Log directory: {log_dir}")
    print("===========================================")
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

    print("===========================================\n")
    
    images_np, c2w_matrices_np, intrinsics_np = training.images, training.poses, training.intrinsics
    images_val_np, c2w_val_np = validation.images, validation.poses
    N_val, H_val, W_val, _ = images_val_np.shape
    splits = {name: scene.split(name).describe() for name in ('train', 'val')}
    check_dataset(checkpoint, splits)
    experiment = experiment_metadata(config, splits, scene_normalization, scene.describe())

    # Create the dataset and DataLoader
    per_image_sampling = reference_baseline and config['no_batching'].lower() == 'true'
    if not per_image_sampling:
        rays = ray_function(images_np, c2w_matrices_np, intrinsics_np, normalize=not reference_baseline)
        dataset = RayDataset(*rays)
        data_loader = DataLoader(dataset, batch_size=num_random_rays, shuffle=True,
                                 num_workers=4 if device.type == 'cuda' else 0,
                                 pin_memory=(device.type == 'cuda'))
        loader_iter = iter(data_loader)

    # Set up the optimizer and loss function
    optimizer = (KerasAdam(model.parameters(), lr=learning_rate) if reference_baseline
                 else optim.Adam(model.parameters(), lr=learning_rate))
    mse_loss = nn.MSELoss()

    # Learning rate scheduler
    gamma = lr_decay_factor ** (1 / (lr_decay * 1000)) if lr_decay > 0 else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: gamma**step if reference_baseline else max(gamma**step, lr_min / learning_rate)
    )

    # TensorBoard writer
    writer_kwargs = {'log_dir': log_dir}
    start_iter = 0
    start_time = datetime.datetime.now()
    if args.resume is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter = get_checkpoint_step(checkpoint)
        print(f"Resuming training from iteration {start_iter}")
        writer_kwargs['purge_step'] = start_iter + 1
    if not 0 <= start_iter <= num_iters:
        raise ValueError('Completed checkpoint updates must be between 0 and num_iters')

    # Restore before compilation, which keeps the optimizer's parameter objects.
    if device.type == 'cuda':
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    else:
        print("Skipping torch.compile: CPU-only environment")

    writer = SummaryWriter(**writer_kwargs)
    writer.add_text('config', str(config))

    # Track completed updates independently of the next iteration, including
    # interruption before the first update or during a later forward pass.
    completed_steps = start_iter
    # Training loop
    try:
        with tqdm(total=num_iters, initial=start_iter, desc="Training", unit="it") as pbar:
            for step in range(start_iter + 1, num_iters + 1):
                if per_image_sampling:
                    image_index = np.random.randint(len(images_np))
                    image = images_np[image_index:image_index + 1]
                    rays = ray_function(image, c2w_matrices_np[image_index:image_index + 1],
                                        intrinsics_np[image_index:image_index + 1], normalize=False)
                    origins, directions, targets = rays[:3]
                    height, width = image.shape[1:3]
                    if step - 1 < int(config['precrop_iters']):
                        fraction = float(config['precrop_frac'])
                        dh, dw = int(height // 2 * fraction), int(width // 2 * fraction)
                        yy, xx = np.meshgrid(np.arange(height // 2 - dh, height // 2 + dh),
                                             np.arange(width // 2 - dw, width // 2 + dw), indexing='ij')
                        candidates = (yy * width + xx).reshape(-1)
                    else:
                        candidates = np.arange(height * width)
                    if num_random_rays > len(candidates):
                        raise ValueError('num_random_rays exceeds the available image/crop pixels')
                    indices = np.random.choice(candidates, size=num_random_rays, replace=False)
                    rays_o_batch = torch.from_numpy(origins[0, indices]).float()
                    rays_d_batch = torch.from_numpy(directions[0, indices]).float()
                    target_rgb_batch = torch.from_numpy(targets[0, indices]).float()
                    viewdirs_batch = torch.from_numpy(rays[3][0, indices]).float() if len(rays) == 4 else None
                else:
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(data_loader)
                        batch = next(loader_iter)
                    rays_o_batch, rays_d_batch, target_rgb_batch = batch[:3]
                    viewdirs_batch = batch[3] if len(batch) == 4 else None
                
                rays_o_batch = rays_o_batch.to(device)
                rays_d_batch = rays_d_batch.to(device)
                target_rgb_batch = target_rgb_batch.to(device)
                
                render_options = ({'num_importance': int(config['num_importance']),
                                   'netchunk': int(config['netchunk']), 'perturb': float(config['perturb']),
                                   'lindisp': config['lindisp'].lower() == 'true',
                                   'raw_noise_std': float(config['raw_noise_std']), 'return_aux': True}
                                  if reference_baseline else {})
                if viewdirs_batch is not None:
                    render_options['view_directions'] = viewdirs_batch.to(device)
                rendered = render_nerf(
                    model,
                    rays_o_batch,
                    rays_d_batch,
                    near,
                    far,
                    num_samples=num_samples,
                    device=device,
                    white_background=scene.white_background,
                    chunk_size=chunk_size,
                    scene_normalization=scene_normalization, **render_options
                )

                # Compute loss, backpropagate, and update model
                optimizer.zero_grad()
                pred_rgb = rendered['rgb_map'] if isinstance(rendered, dict) else rendered
                fine_loss = mse_loss(pred_rgb, target_rgb_batch)
                coarse_loss = (mse_loss(rendered['rgb0'], target_rgb_batch)
                               if isinstance(rendered, dict) and 'rgb0' in rendered else None)
                loss = fine_loss + coarse_loss if coarse_loss is not None else fine_loss
                loss.backward()
                optimizer.step()
                completed_steps = step
                scheduler.step()

                pbar.update(1)

                # Log metrics and write to TensorBoard
                if step == 1 or step % log_interval == 0:
                    log_training_metrics(step, scheduler, fine_loss, start_time, writer)
                    if coarse_loss is not None:
                        writer.add_scalar('loss/coarse', coarse_loss.item(), step)
                        writer.add_scalar('loss/total', loss.item(), step)

                # Save checkpoint
                if step % save_interval == 0 and step < num_iters:
                    model_filename = save_checkpoint(step, 
                                                     model, 
                                                     optimizer, 
                                                     scheduler, 
                                                     save_path, 
                                                     model_type,
                                                     experiment_name,
                                                     scene_normalization=scene_normalization,
                                               experiment=experiment)
                    elapsed_str = format_elapsed_time(start_time)
                    tqdm.write(f"[{elapsed_str}] Model saved to {model_filename} at iteration {step}")

                # Log validation metrics
                if step % val_interval == 0 or (step == 1 and first_step_render):
                    # Select a random image and render it for validation
                    test_image_index = np.random.randint(N_val)
                    single_val_image = images_val_np[test_image_index:test_image_index+1]
                    single_val_c2w = c2w_val_np[test_image_index:test_image_index+1]
                    rays_val = ray_function(single_val_image, single_val_c2w,
                                            validation.intrinsics[test_image_index:test_image_index+1],
                                            normalize=not reference_baseline)
                    rays_o_val_np, rays_d_val_np = rays_val[:2]
                    rays_o_val = torch.from_numpy(rays_o_val_np).float().to(device).squeeze(0)
                    rays_d_val = torch.from_numpy(rays_d_val_np).float().to(device).squeeze(0)
                    val_options = ({'view_directions': torch.from_numpy(rays_val[3]).float().to(device).squeeze(0)}
                                   if len(rays_val) == 4 else {})

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
                            white_background=scene.white_background,
                            chunk_size=chunk_size,
                            stratified=False,
                            scene_normalization=scene_normalization, **val_options,
                            **{key: value for key, value in render_options.items()
                               if key not in ('return_aux', 'view_directions', 'raw_noise_std')}
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

            # Save final model after training is complete
            final_model_path = save_checkpoint(completed_steps,
                                               model, 
                                               optimizer, 
                                               scheduler, 
                                               save_path, 
                                               model_type,
                                               experiment_name,
                                               scene_normalization=scene_normalization,
                                               experiment=experiment)
            elapsed_str = format_elapsed_time(start_time)
            tqdm.write(f"[{elapsed_str}] Training complete!")
            tqdm.write(f"[{elapsed_str}] Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        # Save checkpoint on keyboard interrupt
        elapsed_str = format_elapsed_time(start_time)
        tqdm.write(f"\n[{elapsed_str}] Keyboard interrupt detected! Saving current checkpoint...")
        interrupt_checkpoint_path = save_checkpoint(completed_steps,
                                                    model, 
                                                    optimizer, 
                                                    scheduler, 
                                                    save_path, 
                                                    model_type,
                                                    experiment_name,
                                                    scene_normalization=scene_normalization,
                                               experiment=experiment)
        tqdm.write(f"[{elapsed_str}] Checkpoint saved to {interrupt_checkpoint_path}. Exiting training.")
    finally:
        writer.close()


if __name__ == '__main__':
    main()
