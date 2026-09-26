"""Opt-in CPU smoke checks on the downloaded, real Lego and Fern examples.

Run from the repository root: uv run --locked python scripts/verify_examples.py
This checks execution and finite metrics, not convergence or published scores.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor', type=int, default=32, help='Downsampling factor for CPU smoke checks')
    parser.add_argument('--output', type=Path, default=Path('renders/example_verification'))
    args = parser.parse_args()
    if args.factor < 1:
        parser.error('--factor must be positive')
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    from wavelet_nerf.utils import load_checkpoint, parse_config

    for scene in ('lego', 'fern'):
        if not (repository / 'datasets' / scene).is_dir():
            parser.error('Run bash download_dataset.sh before verifying the examples')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-', dir=output))
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    results = []

    def command(case, stage, arguments):
        print(f'{case}: {stage}', flush=True)
        with (run / f'{case}_{stage}.log').open('w') as log:
            process = subprocess.run([sys.executable, *arguments], cwd=repository,
                                     env=environment, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'{case}: {stage} failed; see {run / (case + "_" + stage + ".log")}')

    for scene in ('lego', 'fern'):
        for model in ('nerf', 'siren', 'wavelet'):
            case = f'{model}_{scene}'
            source = repository / 'config' / f'config_{case}.yaml'
            config = parse_config(source)
            # Retain the example's model architecture and scene conventions.
            # Reduce only resolution, work per update, run length and outputs.
            config.update(dataset_path=str(repository / 'datasets' / scene), dataset_factor=args.factor,
                          half_res=False, experiment_name=case, num_iters=1,
                          log_root=str(run / 'logs'), save_path=str(run / 'models'),
                          num_random_rays=8, num_samples=4, num_samples_eval=4,
                          chunk_size=128,
                          num_render_poses=2, val_interval=1, testskip=8,
                          save_interval=1, log_interval=1)
            if model == 'nerf':
                config.update(netchunk=1024, num_importance=4)
            config_path = run / f'{case}.yaml'

            def write_config():
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))

            write_config()
            command(case, 'train', ['train.py', '--config', str(config_path)])
            checkpoint = run / 'models' / case / f'{case}_000001.pth'
            if load_checkpoint(checkpoint)['step'] != 1:
                raise RuntimeError(f'{case}: first checkpoint does not contain one completed update')
            config['num_iters'] = 2
            write_config()
            command(case, 'resume', ['train.py', '--config', str(config_path), '--resume', str(checkpoint)])
            checkpoint = checkpoint.with_name(f'{case}_000002.pth')
            saved = load_checkpoint(checkpoint)
            if saved['step'] != 2:
                raise RuntimeError(f'{case}: resume did not finish at two completed updates')
            test_output = run / f'{case}_test'
            command(case, 'evaluate', ['eval.py', '--mode', 'test', '--checkpoint', str(checkpoint),
                                       '--output', str(test_output)])
            metrics = json.loads((test_output / 'metrics.json').read_text())
            if scene == 'lego':
                metadata_path = repository / 'datasets/lego/transforms_test.json'
                expected = len(json.loads(metadata_path.read_text())['frames'])
            else:
                metadata_path = repository / 'datasets/fern/poses_bounds.npy'
                expected = len(range(0, len(np.load(metadata_path)), int(config['llff_holdout'])))
            if metrics['summary']['num_images'] != expected or not all(
                    math.isfinite(row['mse']) and math.isfinite(row['psnr_db']) for row in metrics['per_image']):
                raise RuntimeError(f'{case}: expected {expected} complete test views with finite metrics')
            if len(list(test_output.glob('test_*.png'))) != expected:
                raise RuntimeError(f'{case}: test image count differs from the evaluated view count')
            render_output = run / f'{case}_render'
            command(case, 'render', ['eval.py', '--checkpoint', str(checkpoint), '--output', str(render_output)])
            if len(list(render_output.glob('frame_*.png'))) != 2:
                raise RuntimeError(f'{case}: expected two novel-view frames')
            results.append({'case': case, 'source_config': source.relative_to(repository).as_posix(),
                            'resolved_smoke_config': config, 'completed_updates': saved['step'],
                            'test_views': expected, 'render_frames': 2,
                            'dataset_metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                            'test_summary': metrics['summary']})
    report = run / 'verification.json'
    report.write_text(json.dumps({'device': 'cpu', 'purpose': 'execution smoke check; not a quality benchmark',
                                 'cases': results}, indent=2, allow_nan=False) + '\n')
    print(f'All six examples passed. Report: {report}', flush=True)


if __name__ == '__main__':
    main()
