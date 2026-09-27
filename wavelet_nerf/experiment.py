"""Experiment metadata and validation for checkpoint restoration."""

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess

import torch

from wavelet_nerf.run_state import determinism_settings

from wavelet_nerf.configuration import (_COMMON, _MODEL, normalize_config, validate_resolved_config)


_TRAINING = {'seed', 'deterministic', 'num_random_rays', 'num_samples', 'learning_rate',
             'lr_decay', 'lr_decay_factor', 'lr_min'}
_RUNTIME = {'device', 'compile_model', 'num_workers'}


def resolve_experiment_config(config, checkpoint=None, *, training=False):
    """Fill defaults, inherit saved settings, and reject explicit incompatible overrides."""
    config = normalize_config(config)
    saved = normalize_config(checkpoint['experiment']['config']) if checkpoint is not None else None
    if saved is not None:
        if 'model_type' not in saved:
            raise ValueError('Checkpoint config requires model_type')
        required = set(_COMMON) | set(_MODEL[saved['model_type']]) | {'model_type'}
        if required - saved.keys():
            raise ValueError(f'Checkpoint config is incomplete: {sorted(required - saved.keys())}')
        if saved['near'] == 'auto' or saved['far'] == 'auto':
            raise ValueError('Checkpoint config must contain resolved numeric ray bounds')
        validate_resolved_config(saved)
        # Execution choices belong to this invocation, rather than saved hardware.
        saved = {key: value for key, value in saved.items() if key not in _RUNTIME}
    model_type = saved['model_type'] if saved is not None else config.get('model_type', 'nerf')
    model_defaults = dict(_MODEL[model_type])
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
    if model_type == 'nerf':
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
            expected = saved[key]
            if config[key] != expected:
                raise ValueError(f'Checkpoint setting {key!r} is {expected!r}, '
                                 f'but config requests {config[key]!r}')
    resolved = {**defaults, **(saved or {}), **config,
                'model_type': model_type, 'dataset_type': dataset_type}
    for name in ('near', 'far'):
        if resolved[name] == 'auto':
            resolved[name] = defaults[name]
    resolved = validate_resolved_config(resolved, training=training)
    return resolved


def check_dataset(checkpoint, splits):
    """Reject a changed training/validation camera layout on resume."""
    if checkpoint is not None and checkpoint['experiment']['dataset']['splits'] != splits:
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


def experiment_metadata(config, splits, scene_normalization, dataset_metadata, *, device=None):
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
                                    config.get('model_type') == 'nerf' else 'unit world rays'),
            'scene_normalization': scene_normalization.to_dict(),
            'splits': splits,
            **dataset_metadata,
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
