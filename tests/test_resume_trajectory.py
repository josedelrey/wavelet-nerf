import contextlib
import io
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

import train
import eval as evaluate
from wavelet_nerf.utils import load_checkpoint


class ResumeTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        scene = self.root / 'scene'
        scene.mkdir()
        frames = []
        for index in range(2):
            pixels = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3) * (index + 1)
            imageio.imwrite(scene / f'{index}.png', pixels)
            pose = np.eye(4)
            pose[2, 3] = 4
            pose[0, 3] = index * .1
            frames.append({'file_path': f'./{index}', 'transform_matrix': pose.tolist()})
        for split in ('train', 'val'):
            (scene / f'transforms_{split}.json').write_text(json.dumps({'camera_angle_x': .8, 'frames': frames}))
        self.base = {
            'dataset_path': str(scene), 'log_root': str(self.root / 'logs'),
            'save_path': str(self.root / 'models'), 'num_random_rays': 5,
            'num_samples': 3, 'num_samples_eval': 3, 'chunk_size': 1024,
            'val_interval': 2, 'save_interval': 2, 'log_interval': 2,
            'seed': 123, 'deterministic': True, 'device': 'cpu', 'compile_model': False,
        }

    def run_training(self, name, total, settings, resume=None, extra_args=(), interrupt_after=None):
        config = {**self.base, **settings, 'experiment_name': name, 'num_iters': total}
        path = self.root / f'{name}.yaml'
        path.write_text(yaml.safe_dump(config))
        argv = ['train.py', '--config', str(path), *extra_args]
        if resume:
            argv += ['--resume', str(resume)]
        render = train.render_nerf
        calls = 0

        def interrupted_render(*args, **kwargs):
            nonlocal calls
            if kwargs.get('stratified', True):
                if calls == interrupt_after:
                    random.random()
                    np.random.rand()
                    torch.rand(3)
                    raise KeyboardInterrupt
                calls += 1
            return render(*args, **kwargs)

        with patch.object(train, 'render_nerf', interrupted_render), patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            train.main()
        checkpoint = self.root / 'models' / name / f'{name}_{(interrupt_after if interrupt_after is not None else total):06d}.pth'
        return checkpoint

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_uninterrupted_weights_optimizer_rng_and_sampling(self):
        cases = [
            {'model_type': 'siren', 'siren_hidden_dim': 8, 'num_layers': 2},
            {'model_type': 'nerf', 'hidden_dim': 8, 'num_importance': 2,
             'pos_encoding_dim': 2, 'dir_encoding_dim': 1, 'no_batching': True},
        ]
        for index, settings in enumerate(cases):
            with self.subTest(model=settings['model_type']):
                whole = load_checkpoint(self.run_training(f'whole{index}', 7, settings))
                partial = self.run_training(f'split{index}', 3, settings)
                resumed = load_checkpoint(self.run_training(f'split{index}', 7, settings, resume=partial))
                for key in ('model_state_dict', 'optimizer_state_dict',
                            'scheduler_state_dict', 'training_state'):
                    self.assert_nested_equal(whole[key], resumed[key])
                log_dir = self.root / 'logs' / f'split{index}'
                original = yaml.safe_load((log_dir / 'config.yaml').read_text())
                invocation = yaml.safe_load((log_dir / 'config.resume-000003.yaml').read_text())
                self.assertEqual(original['num_iters'], 3)
                self.assertEqual(invocation['num_iters'], 7)
                self.assertIn('lr_decay_factor', original)
                metadata = json.loads((log_dir / 'experiment.json').read_text())
                self.assertTrue(metadata['environment']['determinism']['algorithms'])

    def test_interrupted_forward_resumes_last_completed_random_and_batch_state(self):
        settings = {'model_type': 'siren', 'siren_hidden_dim': 8, 'num_layers': 2}
        whole = load_checkpoint(self.run_training('whole', 7, settings))
        interrupted = self.run_training('interrupted', 7, settings, interrupt_after=3)
        self.assertEqual(load_checkpoint(interrupted)['step'], 3)
        resumed = load_checkpoint(self.run_training('interrupted', 7, settings, resume=interrupted))
        for key in ('model_state_dict', 'optimizer_state_dict', 'training_state'):
            self.assert_nested_equal(whole[key], resumed[key])

    def test_validation_uses_its_own_split_size(self):
        scene = self.root / 'scene'
        metadata = json.loads((scene / 'transforms_train.json').read_text())
        third = {**metadata['frames'][0], 'file_path': './2'}
        pose = np.array(third['transform_matrix'])
        pose[0, 3] = .2
        third['transform_matrix'] = pose.tolist()
        imageio.imwrite(scene / '2.png', np.full((3, 4, 3), 100, dtype=np.uint8))
        for count in (1, 3):
            with self.subTest(validation_views=count):
                frames = metadata['frames'][:1] if count == 1 else [*metadata['frames'], third]
                (scene / 'transforms_val.json').write_text(json.dumps({**metadata, 'frames': frames}))
                settings = {'model_type': 'siren', 'siren_hidden_dim': 8, 'num_layers': 2,
                            'first_step_render': True, 'seed': 3}
                # This seed selects the final validation view in both splits.
                # Using the two-view training bound would fail or omit that view.
                with patch.object(train, 'render_camera', wraps=train.render_camera) as render:
                    path = self.run_training(f'validation{count}', 1, settings)
                render.assert_called_once()
                np.testing.assert_allclose(render.call_args.args[3],
                                           np.array(frames[-1]['transform_matrix']))
                self.assertEqual(load_checkpoint(path)['step'], 1)

    def test_writer_is_closed_when_initial_logging_fails(self):
        settings = {'model_type': 'siren', 'siren_hidden_dim': 8, 'num_layers': 2}
        with patch.object(train, 'SummaryWriter') as factory:
            writer = factory.return_value
            writer.add_text.side_effect = RuntimeError('logging failed')
            with self.assertRaisesRegex(RuntimeError, 'logging failed'):
                self.run_training('failure', 1, settings)
            writer.close.assert_called_once()

    def test_render_overwrite_removes_frames_from_a_longer_previous_render(self):
        settings = {'model_type': 'siren', 'siren_hidden_dim': 8, 'num_layers': 2}
        checkpoint = self.run_training('render', 1, settings)
        output = self.root / 'renders'
        overrides = self.root / 'render-overrides.yaml'

        def render(count, overwrite=False):
            overrides.write_text(yaml.safe_dump({'num_render_poses': count, 'device': 'cpu'}))
            argv = ['eval.py', '--checkpoint', str(checkpoint), '--config', str(overrides),
                    '--output', str(output)]
            if overwrite:
                argv.append('--overwrite')
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                evaluate.main()

        render(3)
        (output / 'notes.txt').write_text('keep')
        with self.assertRaisesRegex(FileExistsError, '--overwrite'):
            render(1)
        render(1, overwrite=True)
        self.assertEqual(len(list(output.glob('frame_*.png'))), 1)
        self.assertTrue((output / 'frame_0000.png').exists())
        self.assertEqual((output / 'notes.txt').read_text(), 'keep')


    def test_new_run_requires_overwrite_and_replaces_old_artifacts(self):
        settings = {'model_type': 'siren', 'siren_hidden_dim': 8, 'num_layers': 2}
        old = self.run_training('same', 3, settings)
        with self.assertRaisesRegex(FileExistsError, '--overwrite'):
            self.run_training('same', 1, settings)
        new = self.run_training('same', 1, settings, extra_args=['--overwrite'])
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        config = yaml.safe_load((self.root / 'logs' / 'same' / 'config.yaml').read_text())
        self.assertEqual(config['num_iters'], 1)


if __name__ == '__main__':
    unittest.main()
