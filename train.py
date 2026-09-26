import os
import argparse
import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from wavelet_nerf.data import (load_configured_scene, resolve_sampling_bounds, PixelRaySampler,
                          RayBatchDataset, PixelBatchPlans)
from wavelet_nerf.model_factory import create_model, model_hyperparameters
from wavelet_nerf.rendering import render_nerf, render_camera
from wavelet_nerf.runtime import add_runtime_arguments, runtime_overrides, resolve_device, prepare_model
from wavelet_nerf.scene import SceneNormalization, resolve_scene_normalization
from wavelet_nerf.nerf_reference import KerasAdam
from wavelet_nerf.experiment import (resolve_experiment_config,
                                check_dataset, experiment_metadata)
from wavelet_nerf.run_state import (capture_rng, restore_rng,
                               configure_reproducibility, prepare_output, write_run_artifacts)
from wavelet_nerf.loss import mse_to_psnr
from wavelet_nerf.utils import parse_config, format_elapsed_time
from wavelet_nerf.utils import load_checkpoint, save_checkpoint, log_training_metrics, get_checkpoint_step


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train NeRF on a given dataset using volumetric rendering."
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to a validated YAML configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint file to resume training from')
    add_runtime_arguments(parser, training=True)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing checkpoints, run metadata and TensorBoard events')
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error('--resume and --overwrite cannot be combined')
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = resolve_experiment_config({**parse_config(args.config), **runtime_overrides(args)}, checkpoint, training=True)
    device = resolve_device(config['device'])
    print(f"Using device: {device}")

    # Reproducibility
    seed = int(config['seed'])
    configure_reproducibility(seed, config['deterministic'])

    # Dataset parameters
    dataset_path = config['dataset_path']
    print('Loading scene...')
    scene = load_configured_scene(config, splits=('train', 'val'))
    validation = scene.split('val')

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
    experiment_name = config.get('experiment_name')
    if not experiment_name:
        raise ValueError("Config must define an 'experiment_name' field")
    log_dir = os.path.join(log_root, experiment_name)

    # Model saving parameters
    checkpoint_root = config['save_path']
    save_path = os.path.join(checkpoint_root, experiment_name)
    save_interval = int(config['save_interval'])

    # Learning rate decay parameters
    lr_decay = float(config['lr_decay'])
    lr_decay_factor = float(config['lr_decay_factor'])
    lr_min = float(config['lr_min'])

    # First step render flag
    first_step_render = config['first_step_render']

    model_type = config['model_type']
    for path in (log_dir, save_path):
        prepare_output(path, ('*.pth', 'events.out.tfevents.*', 'config*.yaml', 'experiment*.json'),
                       overwrite=args.overwrite, resume=args.resume is not None)

    scene_normalization = resolve_scene_normalization(config, checkpoint)
    reference_baseline = model_type == 'nerf' and config['baseline_version'] == 'reference'
    if reference_baseline:
        scene_normalization = SceneNormalization()
    config['scene_center'] = list(scene_normalization.center)
    config['scene_scale'] = scene_normalization.scale

    model = create_model(config).to(device)

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

    for name, value in model_hyperparameters(config).items():
        print(f"{name}: {value}")

    print("===========================================\n")
    
    images_val_np, c2w_val_np = validation.images, validation.poses
    N_val, H_val, W_val, _ = images_val_np.shape
    splits = {name: scene.describe_split(name) for name in ('train', 'val')}
    check_dataset(checkpoint, splits)
    experiment = experiment_metadata(config, splits, scene_normalization, scene.describe(), device=device)
    experiment['ray_sampling'] = {
        'protocol': 'independent_pixel_batches_v1',
        'per_image': reference_baseline and config['no_batching'],
        'replacement_within_batch': False,
    }

    # RGB stays in one scene array; only sampled pixels acquire ray geometry.
    per_image_sampling = reference_baseline and config['no_batching']
    sampler = PixelRaySampler(scene.images, scene.poses, scene.intrinsics,
                              image_indices=scene.splits['train'], dataset_type=scene.dataset_type,
                              normalize=not reference_baseline, seed=seed)
    if checkpoint and checkpoint.get('training_state'):
        sampler.load_state_dict(checkpoint['training_state']['sampler'])

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
    model = prepare_model(model, config, device)

    if checkpoint and not checkpoint.get('training_state'):
        warnings.warn('Checkpoint has no RNG/sampler state; resume cannot reproduce '
                      'the uninterrupted trajectory', stacklevel=2)
    write_run_artifacts(log_dir, experiment, step=start_iter if checkpoint else None)

    # Track completed updates independently of the next iteration, including
    # interruption before the first update or during a later forward pass.
    completed_steps = start_iter
    # Training loop
    boundary_rng = (checkpoint['training_state']['rng'] if checkpoint and checkpoint.get('training_state')
                    else capture_rng())

    def training_state():
        return {'rng': boundary_rng, 'sampler': sampler.state_dict()}

    writer = SummaryWriter(**writer_kwargs)
    try:
        if checkpoint and checkpoint.get('training_state'):
            restore_rng(checkpoint['training_state']['rng'])
        boundary_rng = capture_rng()
        writer.add_text('config', str(config))
        plans = PixelBatchPlans(sampler, num_random_rays, start_iter, num_iters,
                                per_image=per_image_sampling, precrop_iters=config.get('precrop_iters', 0),
                                precrop_frac=config.get('precrop_frac', .5))
        data_loader = DataLoader(RayBatchDataset(sampler), sampler=plans, batch_size=None,
                                 generator=torch.Generator().manual_seed(seed),
                                 num_workers=config['num_workers'], pin_memory=(device.type == 'cuda'))
        loader_iter = iter(data_loader)
        with tqdm(total=num_iters, initial=start_iter, desc="Training", unit="it") as pbar:
            for step in range(start_iter + 1, num_iters + 1):
                batch = next(loader_iter)
                rays_o_batch, rays_d_batch, target_rgb_batch = batch['rays'][:3]
                viewdirs_batch = batch['rays'][3] if len(batch['rays']) == 4 else None

                rays_o_batch = rays_o_batch.to(device)
                rays_d_batch = rays_d_batch.to(device)
                target_rgb_batch = target_rgb_batch.to(device)
                
                render_options = ({'num_importance': int(config['num_importance']),
                                   'perturb': float(config['perturb']),
                                   'lindisp': config['lindisp'],
                                   'raw_noise_std': float(config['raw_noise_std']), 'return_aux': True}
                                  if reference_baseline else {})
                render_options['netchunk'] = config['netchunk']
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
                if sampler is not None:
                    sampler.commit(batch['rng'])
                boundary_rng = capture_rng()

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
                                               experiment=experiment, training_state=training_state())
                    elapsed_str = format_elapsed_time(start_time)
                    tqdm.write(f"[{elapsed_str}] Model saved to {model_filename} at iteration {step}")

                # Log validation metrics
                if step % val_interval == 0 or (step == 1 and first_step_render):
                    # Select a random image and render it for validation
                    test_image_index = np.random.default_rng([seed, step]).integers(N_val)
                    single_val_image = images_val_np[test_image_index:test_image_index+1]
                    single_val_c2w = c2w_val_np[test_image_index:test_image_index+1]
                    tqdm.write("Rendering validation image...")
                    
                    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                        model.eval()
                        with torch.no_grad():
                            pred_val_rgb = render_camera(
                                model, H_val, W_val, single_val_c2w[0],
                                validation.intrinsics[test_image_index], near, far,
                                dataset_type=scene.dataset_type, num_samples=num_samples_eval,
                                device=device, white_background=scene.white_background,
                                chunk_size=chunk_size, scene_normalization=scene_normalization,
                                **{key: value for key, value in render_options.items()
                                   if key not in ('return_aux', 'view_directions', 'raw_noise_std')})
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
                                               experiment=experiment, training_state=training_state())
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
                                               experiment=experiment, training_state=training_state())
        tqdm.write(f"[{elapsed_str}] Checkpoint saved to {interrupt_checkpoint_path}. Exiting training.")
    finally:
        writer.close()


if __name__ == '__main__':
    main()
