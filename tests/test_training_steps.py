import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

import numpy as np
import torch

import train
from modules.datasets import SceneSplit


class TinyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class TrainingStepTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_training(self, name, total, resume=None, interrupt_after=None,
                     first_step_render=False, scene_settings=None,
                     save_key='save_path', save_directory=None):
        checkpoint_root = self.root / 'models' if save_directory is None else Path(save_directory)
        config = {
            'experiment_name': name,
            'log_root': str(self.root / 'logs'),
            save_key: str(checkpoint_root),
            'num_iters': str(total),
            'save_interval': '1',
            'log_interval': '1',
            'val_interval': '2',
            'first_step_render': str(first_step_render),
            'num_random_rays': '1',
        }
        config.update(scene_settings or {})
        images = np.zeros((1, 1, 1, 3), dtype=np.float32)
        poses = np.eye(4, dtype=np.float32)[None]
        split = SceneSplit(images, poses,
                           np.array([[[1., 0., .5], [0., 1., .5], [0., 0., 1.]]], dtype=np.float32),
                           np.array([[2., 6.]], dtype=np.float32), np.array([0]), ('./0',))
        scene = SimpleNamespace(dataset_type='blender', white_background=True,
                                sampling_bounds=(2., 6.), split=lambda name: split,
                                describe=lambda: {})
        training_calls = 0
        scene_transforms = []

        def render(model, rays_o, rays_d, *args, **kwargs):
            nonlocal training_calls
            scene_transforms.append(kwargs['scene_normalization'].to_dict())
            if kwargs.get('stratified', True):
                if training_calls == interrupt_after:
                    raise KeyboardInterrupt
                training_calls += 1
            return torch.sigmoid(model.weight).expand(len(rays_o), 3)

        # Keep real Adam, LambdaLR, checkpoint I/O, and the training loop;
        # replace expensive data/model/rendering and external logging only.
        loader = train.DataLoader

        def cpu_loader(*args, **kwargs):
            return loader(*args, **{**kwargs, 'num_workers': 0})

        argv = ['train.py', '--config', 'unused.txt']
        if resume is not None:
            argv += ['--resume', str(resume)]
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch('sys.argv', argv))
            stack.enter_context(patch.object(train.torch.cuda, 'is_available', return_value=False))
            stack.enter_context(patch.object(train, 'parse_config', return_value=config))
            stack.enter_context(patch.object(train, 'load_configured_scene', return_value=scene))
            stack.enter_context(patch.object(train, 'NeRF', TinyModel))
            stack.enter_context(patch.object(train, 'Siren', TinyModel))
            stack.enter_context(patch.object(train, 'DataLoader', cpu_loader))
            stack.enter_context(patch.object(train, 'render_nerf', render))
            writer_factory = stack.enter_context(patch.object(train, 'SummaryWriter'))
            progress = stack.enter_context(patch.object(train, 'tqdm'))
            train.main()
        return SimpleNamespace(
            folder=checkpoint_root / name,
            training_calls=training_calls,
            writer=writer_factory.return_value,
            writer_kwargs=writer_factory.call_args.kwargs,
            progress=progress.return_value.__enter__.return_value,
            scene_transforms=scene_transforms,
        )

    def checkpoint(self, path, expected):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['format_version'], 2)
        experiment = checkpoint['experiment']
        self.assertEqual(experiment['config']['seed'], '42')
        self.assertEqual(experiment['dataset']['scene_normalization'],
                         checkpoint['scene_normalization'])
        self.assertEqual(experiment['dataset']['splits']['train']['indices'], [0])
        self.assertEqual(checkpoint['step'], expected)
        self.assertEqual(checkpoint['step_semantics'], 'completed_updates')
        self.assertEqual(checkpoint['scheduler_state_dict']['last_epoch'], expected)
        for state in checkpoint['optimizer_state_dict']['state'].values():
            self.assertEqual(int(state['step']), expected)
        return checkpoint

    def test_periodic_final_metrics_and_progress_count_completed_updates(self):
        result = self.run_training('fresh', 3, first_step_render=True)
        self.assertEqual(result.training_calls, 3)
        for step in (1, 2, 3):
            self.checkpoint(result.folder / f'fresh_{step:06d}.pth', step)
        loss_steps = [call.args[2] for call in result.writer.add_scalar.call_args_list
                      if call.args[0] == 'loss']
        self.assertEqual(loss_steps, [1, 2, 3])
        validation_steps = [call.args[2] for call in result.writer.add_scalar.call_args_list
                            if call.args[0] == 'val/psnr']
        self.assertEqual(validation_steps, [1, 2])
        self.assertEqual(result.progress.update.call_args_list, [call(1)] * 3)
        result.writer.close.assert_called_once()

    def test_custom_checkpoint_root_and_legacy_alias_save_and_resume(self):
        for key in ('save_path', 'save_root'):
            with self.subTest(key=key):
                destination = self.root / f'custom_{key}'
                self.run_training(key, 1, save_key=key, save_directory=destination)
                path = destination / key / f'{key}_000001.pth'
                checkpoint = self.checkpoint(path, 1)
                settings = checkpoint['experiment']['config']
                self.assertEqual(settings['save_path'], str(destination))
                self.assertNotIn('save_root', settings)
                self.assertFalse((self.root / 'models' / key).exists())
                resumed_root = self.root / f'relocated_{key}'
                resumed = self.run_training(f'{key}_resumed', 2, resume=path,
                                            save_key=key, save_directory=resumed_root)
                self.checkpoint(resumed.folder / f'{key}_resumed_000002.pth', 2)

    def test_resume_performs_only_remaining_updates(self):
        original = self.run_training('original', 2)
        result = self.run_training('resumed', 3, original.folder / 'original_000002.pth')
        self.assertEqual(result.training_calls, 1)
        self.checkpoint(result.folder / 'resumed_000003.pth', 3)
        self.assertEqual(result.writer_kwargs['purge_step'], 3)

    def test_training_validation_and_resume_use_saved_scene_coordinates(self):
        expected = {'center': [1.0, -2.0, 3.0], 'scale': 4.0}
        original = self.run_training(
            'scene', 2, first_step_render=True,
            scene_settings={'model_type': 'siren', 'scene_center': '1, -2, 3', 'scene_scale': '4'},
        )
        path = original.folder / 'scene_000002.pth'
        self.assertEqual(self.checkpoint(path, 2)['scene_normalization'], expected)
        self.assertTrue(original.scene_transforms)
        self.assertTrue(all(transform == expected for transform in original.scene_transforms))
        resumed = self.run_training(
            'scene_resumed', 4, path,
            scene_settings={'model_type': 'siren', 'scene_center': '99, 99, 99', 'scene_scale': '100'},
        )
        self.assertTrue(all(transform == expected for transform in resumed.scene_transforms))
        self.assertEqual(
            self.checkpoint(resumed.folder / 'scene_resumed_000004.pth', 4)['scene_normalization'],
            expected,
        )

    def test_legacy_periodic_checkpoint_uses_scheduler_update_count(self):
        original = self.run_training('original', 3)
        legacy = self.checkpoint(original.folder / 'original_000002.pth', 2)
        legacy.pop('step_semantics')
        legacy['step'] = 1  # Old periodic saves stored the zero-based loop index.
        path = self.root / 'legacy.pth'
        torch.save(legacy, path)
        result = self.run_training('legacy_resumed', 3, path)
        self.assertEqual(result.training_calls, 1)
        self.checkpoint(result.folder / 'legacy_resumed_000003.pth', 3)

    def test_legacy_final_checkpoint_does_not_add_an_update(self):
        original = self.run_training('original', 3)
        legacy = self.checkpoint(original.folder / 'original_000003.pth', 3)
        legacy.pop('step_semantics')
        path = self.root / 'legacy_final.pth'
        torch.save(legacy, path)
        result = self.run_training('finished', 3, path)
        self.assertEqual(result.training_calls, 0)
        self.checkpoint(result.folder / 'finished_000003.pth', 3)

    def test_interrupt_saves_only_completed_updates_and_resumes(self):
        interrupted = self.run_training('interrupted', 4, interrupt_after=2)
        path = interrupted.folder / 'interrupted_000002.pth'
        self.checkpoint(path, 2)
        interrupted.writer.close.assert_called_once()
        result = self.run_training('continued', 4, path)
        self.assertEqual(result.training_calls, 2)
        self.checkpoint(result.folder / 'continued_000004.pth', 4)

    def test_interrupt_before_first_update_saves_zero(self):
        result = self.run_training('interrupted', 3, interrupt_after=0)
        self.assertEqual(result.training_calls, 0)
        self.checkpoint(result.folder / 'interrupted_000000.pth', 0)


if __name__ == '__main__':
    unittest.main()
