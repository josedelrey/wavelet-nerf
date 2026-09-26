import numpy as np
import argparse
import os
import csv
import json
import math
import imageio
from tqdm import tqdm

from wavelet_nerf.data import load_configured_scene
from wavelet_nerf.model_factory import create_model
from wavelet_nerf.rendering import render_camera
from wavelet_nerf.runtime import add_runtime_arguments, runtime_overrides, resolve_device, prepare_model
from wavelet_nerf.scene import SceneNormalization, resolve_scene_normalization
from wavelet_nerf.utils import load_checkpoint, parse_config
from wavelet_nerf.camera import configured_render_path, render_camera_path
from wavelet_nerf.experiment import resolve_experiment_config
from wavelet_nerf.run_state import configure_reproducibility, prepare_output


def image_metrics(prediction, target):
    """Measure floating-point RGB on [0, 1], before PNG clipping/quantization."""
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 3 or target.size == 0:
        raise ValueError('Prediction and target must have matching H x W x 3 shapes')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Cannot evaluate nonfinite image values')
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    return {'mse': mse, 'psnr_db': -10 * math.log10(mse) if mse > 0 else math.inf}


def write_metrics(rows, output_dir, settings):
    """Write both mean per-view PSNR and PSNR of pooled RGB squared errors."""
    if not rows:
        raise ValueError('Cannot aggregate an empty test split')
    total_values = sum(row['height'] * row['width'] * 3 for row in rows)
    pooled_mse = sum(row['mse'] * row['height'] * row['width'] * 3 for row in rows) / total_values
    summary = {
        'num_images': len(rows),
        'mean_psnr_db': sum(row['psnr_db'] for row in rows) / len(rows),
        'pooled_mse': pooled_mse,
        'pooled_psnr_db': -10 * math.log10(pooled_mse) if pooled_mse > 0 else math.inf,
    }
    report = {'settings': settings, 'summary': summary, 'per_image': rows}

    def json_values(value):
        # Preserve exact-match PSNR without emitting nonstandard JSON tokens.
        if isinstance(value, dict):
            return {key: json_values(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_values(item) for item in value]
        return 'Infinity' if isinstance(value, float) and math.isinf(value) else value

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as file:
        json.dump(json_values(report), file, indent=2, allow_nan=False)
        file.write('\n')
    with open(os.path.join(output_dir, 'metrics.csv'), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate test views or render a novel-view trajectory."
    )
    parser.add_argument('--config', type=str,
                        help='YAML overrides; required for legacy checkpoints')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='rendered_frames',
                        help='Path to output directory')
    parser.add_argument('--mode', choices=('render', 'test'), default='render',
                        help='Render the scene camera path (default), or score every ground-truth test view')
    parser.add_argument('--dataset-path', type=str,
                        help='Override the dataset location for test evaluation')
    add_runtime_arguments(parser)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace generated frames and metrics in an existing output directory')
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    if args.config is None and 'experiment' not in checkpoint:
        parser.error('--config is required for legacy checkpoints')
    overrides = parse_config(args.config) if args.config else {}
    overrides.update(runtime_overrides(args))
    if args.dataset_path is not None:
        overrides['dataset_path'] = args.dataset_path
    config = resolve_experiment_config(overrides, checkpoint)
    device = resolve_device(config['device'])
    print(f"Using device: {device}")

    # Reproducibility
    seed = int(config['seed'])
    configure_reproducibility(seed, config['deterministic'])

    # Parameters
    dataset_path = config['dataset_path']
    model_type = config['model_type']
    reference_baseline = model_type == 'nerf' and config['baseline_version'] == 'reference'
    model_path = args.checkpoint
    output_dir = args.output
    near = float(config['near'])
    far = float(config['far'])
    scene_normalization = resolve_scene_normalization(config, checkpoint)
    if reference_baseline:
        scene_normalization = SceneNormalization()
    num_samples = int(config['num_samples_eval'])
    chunk_size = int(config['chunk_size'])
    num_render_poses = int(config['num_render_poses'])
    if num_samples <= 0 or chunk_size <= 0:
        parser.error('num_samples_eval and chunk_size must be positive')
    if args.mode == 'render' and num_render_poses <= 0:
        parser.error('num_render_poses must be positive in render mode')

    print("===== Evaluation Configuration Summary =====")
    print(f"Dataset path: {dataset_path}")
    print(f"Model type: {model_type}")
    print(f"Model path: {model_path}")
    print(f"Mode: {args.mode}")
    print(f"Log directory: {output_dir}")
    print(f"Near: {near}")
    print(f"Far: {far}")
    print(f"Scene center: {scene_normalization.center}")
    print(f"Scene scale: {scene_normalization.scale}")
    print(f"Num samples: {num_samples}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of render poses: {num_render_poses}")
    print("=============================================")

    model = create_model(config).to(device)

    # Load the model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    model = prepare_model(model, config, device)
    
    # Test views use loaded cameras. Novel paths can render solely from saved metadata.
    dataset_metadata = checkpoint.get('experiment', {}).get('dataset', {})
    white_background = config['white_background']
    if args.mode == 'test':
        scene = load_configured_scene({**config, 'testskip': '1'}, splits=('test',))
        split = scene.split('test')
        images, render_poses, intrinsics = split.images, split.poses, split.intrinsics
        image_paths, source_indices = split.frame_paths, split.indices
        height, width = images.shape[1:3]
        if 'world_to_scene' in dataset_metadata and not np.allclose(
                scene.world_to_scene, dataset_metadata['world_to_scene']):
            raise ValueError('Evaluation scene preprocessing differs from the saved experiment')
    elif 'experiment' in checkpoint:
        camera = dataset_metadata.get('render_intrinsics')
        if camera is None:
            # Checkpoints written before explicit scene metadata stored one camera.
            camera = dataset_metadata['splits']['train']['intrinsics']
            camera = {**camera, 'matrix': [[camera['fx'], 0, camera.get('cx', camera['width'] / 2)],
                                         [0, camera.get('fy', camera['fx']), camera.get('cy', camera['height'] / 2)],
                                         [0, 0, 1]]}
        height, width = camera['height'], camera['width']
        intrinsics = np.asarray(camera['matrix'], dtype=np.float32)[None]
        path_settings = dataset_metadata.get('render_path')
        if path_settings is None:
            if config['dataset_type'] == 'llff':
                raise ValueError('LLFF checkpoint is missing saved spiral settings; '
                                 'use a checkpoint with scene/render-path metadata')
            path_settings = {'type': 'orbit', 'elevation': -30., 'radius': 4.}
    else:
        scene = load_configured_scene(config, splits=('test',))
        height, width = scene.images.shape[1:3]
        intrinsics = scene.intrinsics[:1]
        path_settings = scene.render_path
    if args.mode == 'render':
        render_poses = render_camera_path(configured_render_path(path_settings, config), num_render_poses)

    prepare_output(output_dir, ('frame_*.png', 'test_*.png', 'metrics.json', 'metrics.csv'),
                   overwrite=args.overwrite)

    # Initialize tqdm for the rendering loop
    render_loop = tqdm(
        range(len(render_poses)),
        desc="Evaluating test views" if args.mode == 'test' else "Rendering frames",
        unit="frame",
        dynamic_ncols=True
    )
        
    # Render the images
    model.eval()
    rows = []
    for i in render_loop:
        camera = intrinsics[i:i + 1] if args.mode == 'test' else intrinsics
        pred_val_rgb = render_camera(
            model, height, width, render_poses[i], camera[0], near, far,
            dataset_type=config['dataset_type'], num_samples=num_samples, device=device,
            white_background=white_background, chunk_size=chunk_size,
            scene_normalization=scene_normalization, netchunk=config['netchunk'],
            **({'num_importance': int(config['num_importance']),
                'lindisp': config['lindisp']} if reference_baseline else {}))

        # Reshape to image
        H_val, W_val = height, width
        pred_val_rgb = pred_val_rgb.reshape(H_val, W_val, 3).cpu().numpy()

        if args.mode == 'test':
            rows.append({
                'index': int(source_indices[i]), 'file_path': image_paths[i],
                'height': H_val, 'width': W_val,
                **image_metrics(pred_val_rgb, images[i]),
            })
        
        # Quantization is for visualization only, never metric computation.
        pred_val_rgb_clamped = np.clip(pred_val_rgb, 0.0, 1.0)
        frame = (pred_val_rgb_clamped * 255).astype(np.uint8)

        # Save frame as PNG
        prefix = 'test' if args.mode == 'test' else 'frame'
        frame_filename = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        imageio.imwrite(frame_filename, frame)

    if args.mode == 'test':
        summary = write_metrics(rows, output_dir, {
            'checkpoint': os.path.abspath(model_path), 'model_type': model_type,
            'dataset_path': os.path.abspath(dataset_path), 'split': 'test',
            'data_range': 1.0, 'color_space': 'stored RGB; no linear-light conversion',
            'background': 'white' if white_background else 'black',
            'metric_input': 'unclipped, unquantized float RGB; full image; no mask or crop',
            'mse_dtype': 'float64', 'sampling': 'uniform; stratified=False',
            'dataset_type': config['dataset_type'],
            'split_protocol': scene.split_protocol,
            'frame_order': 'sorted LLFF images, holdout subset' if config['dataset_type'] == 'llff'
                           else 'transforms_test.json frames',
            'intrinsics': intrinsics.tolist(),
            'num_samples': num_samples, 'chunk_size': chunk_size,
            'near': near, 'far': far, 'scene_normalization': scene_normalization.to_dict(),
            'seed': seed,
            'baseline_version': config.get('baseline_version'),
            'num_importance': int(config['num_importance']) if reference_baseline else 0,
        })
        print(f"Test views: {summary['num_images']}; mean PSNR: {summary['mean_psnr_db']:.4f} dB; "
              f"pooled PSNR: {summary['pooled_psnr_db']:.4f} dB")
        print(f"Metrics saved to {output_dir}/metrics.json and metrics.csv")


if __name__ == '__main__':
    main()
