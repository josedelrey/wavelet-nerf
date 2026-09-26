"""Plain checkpoint metadata and compatibility checks for experiment restoration."""

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import warnings


_COMMON = {
    'dataset_path': './datasets/lego', 'seed': 42,
    'dataset_type': 'blender', 'dataset_factor': 1, 'llff_holdout': 8,
    'llff_bounds_scale': 0.75, 'llff_recenter': True, 'white_background': True,
    'near': 2.0, 'far': 6.0, 'num_random_rays': 1024,
    'num_samples': 256, 'num_samples_eval': 256, 'chunk_size': 8192,
    'learning_rate': 5e-4, 'lr_decay': 150.0, 'lr_decay_factor': 0.1,
    'lr_min': 1e-5, 'num_iters': 150000, 'save_interval': 5000,
    'log_interval': 10, 'val_interval': 1000, 'first_step_render': False,
    'log_root': './logs', 'save_path': './models', 'num_render_poses': 40,
}
_MODEL = {
    'nerf': {
        'pos_encoding_dim': 10, 'dir_encoding_dim': 4, 'hidden_dim': 256,
        'baseline_version': 'reference', 'num_importance': 128, 'netchunk': 65536,
        'perturb': 1.0, 'lindisp': False, 'raw_noise_std': 0.0,
        'white_background': True, 'no_batching': True, 'precrop_iters': 0,
        'precrop_frac': 0.5, 'half_res': False, 'testskip': 8,
    },
    'siren': {
        'num_layers': 8, 'siren_hidden_dim': 256, 'siren_dir_encoding_dim': 4,
        'sigma_mul': 10.0, 'rgb_mul': 1.0, 'w0': 30.0, 'hidden_w0': 1.0,
    },
    'wavelet': {
        'wave_in_features': 3, 'wave_hidden_dim': 256, 'wave_num_layers': 8,
        'wave_dir_encoding_dim': 4, 'input_scale': 256.0, 'weight_scale': 1.0,
        'alpha': 6.0, 'beta': 0.5, 'omega0': 5.0, 'normalized': True,
    },
}
_TRAINING = {'seed', 'num_random_rays', 'num_samples', 'learning_rate',
             'lr_decay', 'lr_decay_factor', 'lr_min'}


def _canonicalize_save_path(config):
    """Accept the former config name while storing only canonical save_path."""
    config = dict(config)
    if 'save_root' in config:
        root = config.pop('save_root')
        if 'save_path' in config and Path(config['save_path']).resolve() != Path(root).resolve():
            raise ValueError('save_root and save_path specify different checkpoint directories; '
                             'use save_path only')
        config.setdefault('save_path', root)
    if 'save_path' in config:
        if not str(config['save_path']).strip():
            raise ValueError('save_path must specify a nonempty checkpoint directory')
        config['save_path'] = str(config['save_path'])
    return config


def resolve_experiment_config(config, checkpoint=None, *, training=False):
    """Fill defaults, inherit saved settings, and reject explicit incompatible overrides."""
    config = _canonicalize_save_path(config)
    saved = checkpoint.get('experiment', {}).get('config') if checkpoint else None
    if saved is not None:
        saved = dict(saved)
        # Old checkpoints can contain both an ignored save_root and the actual
        # resolved save_path. Preserve their recorded destination on restoration.
        if 'save_path' in saved:
            saved.pop('save_root', None)
        saved = _canonicalize_save_path(saved)
    if checkpoint is not None and saved is None:
        warnings.warn('Legacy checkpoint has no experiment metadata; compatibility '
                      'cannot be verified. Supply the original config.', stacklevel=2)
    model_type = str((saved or {}).get('model_type',
                     (checkpoint or {}).get('model_type', config.get('model_type', 'nerf')))).lower()
    if model_type == 'multiscalewavelet':
        model_type = 'wavelet'
    if model_type not in _MODEL:
        raise ValueError(f'Invalid model type: {model_type}')
    model_defaults = dict(_MODEL[model_type])
    if model_type == 'nerf':
        # Old checkpoint architecture cannot be converted into two reference MLPs.
        legacy = any(key.startswith('block1.') for key in
                     (checkpoint or {}).get('model_state_dict', {}))
        model_defaults['baseline_version'] = 'legacy' if legacy else 'reference'
    defaults = {**_COMMON, **model_defaults}
    dataset_type = str((saved or {}).get('dataset_type', config.get('dataset_type', 'blender'))).lower()
    if dataset_type not in ('blender', 'llff'):
        raise ValueError('dataset_type must be blender or llff')
    defaults['dataset_type'] = dataset_type
    if dataset_type == 'llff':
        defaults.update(near=0.0, far=1.0, dataset_factor=8,
                        white_background=False, dataset_path='./datasets/fern', num_render_poses=120)
        if model_type == 'nerf':
            defaults.update(white_background=False, no_batching=False,
                            num_importance=64, raw_noise_std=1.0)
    if model_type == 'nerf' and model_defaults['baseline_version'] == 'reference':
        defaults.update(num_samples=64, num_samples_eval=64, lr_decay=500.,
                        lr_min=0., num_iters=500000)
        if dataset_type == 'llff':
            # Published forward-facing LLFF settings differ from Blender.
            defaults.update(dataset_factor=4, num_random_rays=4096, lr_decay=250.,
                            raw_noise_std=1., num_iters=200000)
    protected = {'model_type', 'dataset_type', 'near', 'far', 'dataset_factor', 'llff_holdout',
                 'llff_bounds_scale', 'llff_recenter', 'white_background', *_MODEL[model_type]}
    if training:
        protected |= _TRAINING
    protected.discard('netchunk')  # Query batching changes memory use, not the experiment.
    if saved:
        for key in protected & config.keys():
            value = config[key]
            if key == 'model_type':
                value = str(value).lower()
                if value == 'multiscalewavelet':
                    value = 'wavelet'
                expected = saved[key]
            else:
                cast = type(defaults[key])
                value = (str(value).lower() in ('true', '1', 'yes')
                         if cast is bool else cast(value))
                expected = (str(saved.get(key, defaults[key])).lower() in ('true', '1', 'yes')
                            if cast is bool else cast(saved.get(key, defaults[key])))
            if value != expected:
                raise ValueError(f'Checkpoint setting {key!r} is {saved.get(key, defaults.get(key))!r}, '
                                 f'but config requests {config[key]!r}')
    # Keep the existing config-file representation used by the CLI constructors.
    resolved = {**{key: str(value) for key, value in defaults.items()},
                **(saved or {}), **config, 'model_type': model_type, 'dataset_type': dataset_type}
    if dataset_type == 'llff':
        for name, value in (('near', '0.0'), ('far', '1.0')):
            if resolved[name] == 'auto':
                resolved[name] = value
        if float(resolved['near']) != 0 or float(resolved['far']) != 1:
            raise ValueError('LLFF NDC sampling requires near = 0 and far = 1')
        if str(resolved.get('lindisp', 'false')).lower() in ('true', '1', 'yes'):
            raise ValueError('LLFF NDC uses linear depth sampling, not inverse depth')
        if str(resolved.get('white_background', 'false')).lower() in ('true', '1', 'yes'):
            raise ValueError('LLFF reference rendering requires white_background = false')
    if model_type == 'nerf':
        version = resolved['baseline_version']
        if version not in ('reference', 'legacy'):
            raise ValueError('baseline_version must be reference or legacy')
        if checkpoint and legacy and version != 'legacy':
            raise ValueError('Legacy NeRF weights cannot be loaded into the reference coarse/fine baseline; retrain')
        if version == 'reference' and training and float(resolved['lr_min']) != 0:
            raise ValueError('Reference NeRF uses exponential decay without an lr_min floor; set lr_min = 0')
        if version == 'reference':
            if not 0 < float(resolved['precrop_frac']) <= 1 or int(resolved['precrop_iters']) < 0:
                raise ValueError('Reference crop needs 0 < precrop_frac <= 1 and precrop_iters >= 0')
            if float(resolved['raw_noise_std']) < 0 or float(resolved['perturb']) < 0:
                raise ValueError('Density noise and perturb must be nonnegative')
    return resolved


def describe_split(images, poses, focal, indices=None):
    """Record the actual loader output, in transforms_<split>.json frame order."""
    height, width = images.shape[1:3]
    return {
        'indices': list(range(len(images))) if indices is None else indices.tolist(),
        'camera_to_world': poses.tolist(),
        'intrinsics': {'height': int(height), 'width': int(width),
                       'fx': float(focal), 'fy': float(focal),
                       'cx': width / 2, 'cy': height / 2},
    }


def check_dataset(checkpoint, splits):
    """Reject a changed training/validation camera layout on resume."""
    saved = checkpoint.get('experiment') if checkpoint else None
    if saved and saved['dataset']['splits'] != splits:
        raise ValueError('Checkpoint dataset cameras, intrinsics, or split indices '
                         'do not match the loaded dataset')


def experiment_metadata(config, splits, scene_normalization, dataset_metadata=None):
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    lock = root / 'uv.lock'
    return {
        'config': dict(config),
        'dataset': {
            'type': config.get('dataset_type', 'blender'),
            'background': 'black' if config.get('dataset_type') == 'llff' else 'white',
            'ray_space': 'ndc' if config.get('dataset_type') == 'llff' else 'world',
            'ndc_near_plane': 1.0 if config.get('dataset_type') == 'llff' else None,
            'camera_convention': 'integer pixels; center (W/2, H/2); camera -Z forward, +Y up; '
                                 + ('unnormalized NDC geometry rays, unit world viewdirs' if
                                    config.get('dataset_type') == 'llff' else
                                    'unnormalized geometry rays, unit viewdirs' if
                                    config.get('baseline_version') == 'reference' else 'unit world rays'),
            'scene_normalization': scene_normalization.to_dict(),
            'splits': splits,
            **(dataset_metadata or {}),
        },
        'code': {'revision': revision, 'dirty': dirty},
        'environment': {
            'python': platform.python_version(),
            'packages': {name: version(name) for name in
                         ('torch', 'numpy', 'Pillow', 'imageio', 'tensorboard', 'tqdm')},
            'uv_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        },
    }
