"""Plain checkpoint metadata and compatibility checks for experiment restoration."""

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import warnings

import torch

from modules.run_state import determinism_settings

from modules.configuration import (_COMMON, _MODEL, normalize_config, validate_resolved_config)


_TRAINING = {'seed', 'deterministic', 'num_random_rays', 'num_samples', 'learning_rate',
             'lr_decay', 'lr_decay_factor', 'lr_min'}
_RUNTIME = {'device', 'compile_model', 'num_workers'}


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
    config = _canonicalize_save_path(normalize_config(config))
    saved = checkpoint.get('experiment', {}).get('config') if checkpoint else None
    if saved is not None:
        saved = dict(saved)
        # Old checkpoints can contain both an ignored save_root and the actual
        # resolved save_path. Preserve their recorded destination on restoration.
        if 'save_path' in saved:
            saved.pop('save_root', None)
        saved = _canonicalize_save_path(normalize_config(saved, legacy=True))
        # Execution choices belong to this invocation, rather than the saved hardware.
        saved = {key: value for key, value in saved.items() if key not in _RUNTIME}
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
    if saved:
        # Old smoke/config dictionaries sometimes recorded unused options for
        # other models. They did not affect construction; omit them on restore.
        model_keys = set().union(*_MODEL.values())
        saved = {key: value for key, value in saved.items()
                 if key not in model_keys or key in _MODEL[model_type] or key in _COMMON}
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
                            num_importance=128, raw_noise_std=1., num_iters=200000)
    for name in ('near', 'far'):
        if config.get(name) == 'auto':
            config[name] = defaults[name]
    protected = {'model_type', 'dataset_type', 'near', 'far', 'dataset_factor', 'llff_holdout',
                 'llff_bounds_scale', 'llff_recenter', 'white_background', *_MODEL[model_type]}
    if training:
        protected |= _TRAINING
    protected.discard('netchunk')  # Query batching changes memory use, not the experiment.
    if saved:
        for key in protected & config.keys():
            expected = saved.get(key, model_type if key == 'model_type' else defaults.get(key))
            if config[key] != expected:
                raise ValueError(f'Checkpoint setting {key!r} is {expected!r}, '
                                 f'but config requests {config[key]!r}')
    resolved = {**defaults, **(saved or {}), **config,
                'model_type': model_type, 'dataset_type': dataset_type}
    for name in ('near', 'far'):
        if resolved[name] == 'auto':
            resolved[name] = defaults[name]
    if model_type == 'nerf' and checkpoint and legacy and resolved['baseline_version'] != 'legacy':
        raise ValueError('Legacy NeRF weights cannot be loaded into the reference coarse/fine baseline; retrain')
    resolved = validate_resolved_config(resolved, training=training)
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


def accelerator_metadata(device=None):
    """Record the installed build and visible hardware, using plain checkpoint values."""
    cuda_available = torch.cuda.is_available()
    gpus = []
    driver = None
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append({'name': properties.name, 'memory_bytes': properties.total_memory,
                         'compute_capability': [properties.major, properties.minor]})
        try:
            driver = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).splitlines()[0].strip() or None
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return {
        'torch_build': str(torch.__version__),
        'cuda_runtime': torch.version.cuda,
        'device': str(device) if device is not None else ('cuda' if cuda_available else 'cpu'),
        'gpus': gpus,
        'nvidia_driver': driver,
    }


def experiment_metadata(config, splits, scene_normalization, dataset_metadata=None, *, device=None):
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
            'determinism': determinism_settings(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            **accelerator_metadata(device),
            'packages': {name: version(name) for name in
                         ('torch', 'numpy', 'Pillow', 'imageio', 'tensorboard', 'tqdm')},
            'uv_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        },
    }
