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
                     first_step_render=False):
        config = {
            'experiment_name': name,
            'log_root': str(self.root / 'logs'),
            'save_path': str(self.root / 'models'),
            'num_iters': str(total),
            'save_interval': '1',
            'log_interval': '1',
            'val_interval': '2',
            'first_step_render': str(first_step_render),
            'num_random_rays': '1',
        }
        images = np.zeros((1, 1, 1, 3), dtype=np.float32)
        poses = np.eye(4, dtype=np.float32)[None]
        training_calls = 0

        def render(model, rays_o, rays_d, *args, **kwargs):
            nonlocal training_calls
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
            stack.enter_context(patch.object(train, 'load_dataset', return_value=(images, poses, 1.0)))
            stack.enter_context(patch.object(train, 'NeRF', TinyModel))
            stack.enter_context(patch.object(train, 'DataLoader', cpu_loader))
            stack.enter_context(patch.object(train, 'render_nerf', render))
            writer_factory = stack.enter_context(patch.object(train, 'SummaryWriter'))
            progress = stack.enter_context(patch.object(train, 'tqdm'))
            train.main()
        return SimpleNamespace(
            folder=self.root / 'models' / name,
            training_calls=training_calls,
            writer=writer_factory.return_value,
            writer_kwargs=writer_factory.call_args.kwargs,
            progress=progress.return_value.__enter__.return_value,
        )

    def checkpoint(self, path, expected):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
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

    def test_resume_performs_only_remaining_updates(self):
        original = self.run_training('original', 2)
        result = self.run_training('resumed', 3, original.folder / 'original_000002.pth')
        self.assertEqual(result.training_calls, 1)
        self.checkpoint(result.folder / 'resumed_000003.pth', 3)
        self.assertEqual(result.writer_kwargs['purge_step'], 3)

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
