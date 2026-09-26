"""Run artifacts, reproducibility settings, and sampling state."""

import json
import os
from pathlib import Path
import random
import tempfile
import warnings

import numpy as np
import torch
import yaml


def atomic_write(path, write):
    """Publish a complete file, leaving the previous file intact on failure."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_output(path, patterns, *, overwrite=False, resume=False):
    """Require an explicit run mode; overwrite removes only recognized artifacts."""
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not (overwrite or resume):
        raise FileExistsError(f'{path} is not empty; use --resume or --overwrite, '
                              'or choose a new experiment/output directory')
    if overwrite:
        for pattern in patterns:
            for file in path.glob(pattern):
                if file.is_file() or file.is_symlink():
                    file.unlink()
    path.mkdir(parents=True, exist_ok=True)


def write_run_artifacts(log_dir, experiment, *, step):
    """Write effective settings separately from the richer experiment record."""
    config = yaml.safe_dump(experiment['config'], sort_keys=True)
    metadata = json.dumps(experiment, indent=2, allow_nan=False) + '\n'
    # A resumed run keeps its earlier metadata and records the new invocation.
    suffix = '' if step is None else f'.resume-{step:06d}'
    atomic_write(Path(log_dir) / f'config{suffix}.yaml', lambda file: file.write(config.encode()))
    atomic_write(Path(log_dir) / f'experiment{suffix}.json', lambda file: file.write(metadata.encode()))


def configure_reproducibility(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def determinism_settings():
    return {
        'algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }


def capture_rng():
    numpy = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [numpy[0], numpy[1].tolist(), *numpy[2:]],
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if torch.cuda.is_initialized() and len(state['cuda']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state['cuda'])
        else:
            warnings.warn('CUDA RNG state cannot be restored on this device configuration; '
                          'resume will not reproduce the original trajectory', stacklevel=2)
