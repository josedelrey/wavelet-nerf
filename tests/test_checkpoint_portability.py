from pathlib import Path
import tempfile
import unittest

import torch

from wavelet_nerf.utils import load_checkpoint, save_checkpoint


class CheckpointPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.manual_seed(42)
        self.inputs = torch.randn(4, 3)

    def model(self):
        # BatchNorm exercises state-dict metadata as well as weights and buffers.
        return torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2),
        )

    def update(self, model, optimizer, scheduler):
        model.train()
        optimizer.zero_grad()
        model(self.inputs).square().mean().backward()
        optimizer.step()
        scheduler.step()

    def round_trip(self, source_compiled, destination_compiled, legacy=False):
        source = self.model()
        optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        if source_compiled:
            source = torch.compile(source, backend='eager')
        self.update(source, optimizer, scheduler)
        path = save_checkpoint(1, source, optimizer, scheduler, self.root, 'test', 'run')

        if legacy:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            checkpoint['model_state_dict'] = source.state_dict()
            torch.save(checkpoint, path)
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            self.assertFalse(any(key.startswith('_orig_mod.')
                                 for key in checkpoint['model_state_dict']))

        checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint['step'], 1)
        self.assertEqual(checkpoint['model_type'], 'test')
        destination = self.model()
        self.assertEqual(checkpoint['model_state_dict']._metadata,
                         destination.state_dict()._metadata)
        destination.load_state_dict(checkpoint['model_state_dict'])
        restored_optimizer = torch.optim.Adam(destination.parameters(), lr=0.01)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(
            restored_optimizer, step_size=1, gamma=0.9
        )
        restored_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restored_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if destination_compiled:
            destination = torch.compile(destination, backend='eager')

        source.eval()
        destination.eval()
        with torch.no_grad():
            torch.testing.assert_close(source(self.inputs), destination(self.inputs))
        self.update(source, optimizer, scheduler)
        self.update(destination, restored_optimizer, restored_scheduler)
        for expected, actual in zip(source.parameters(), destination.parameters()):
            torch.testing.assert_close(expected, actual)
        self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())
        for state in restored_optimizer.state.values():
            self.assertEqual(int(state['step']), 2)

    def test_new_checkpoints_work_with_all_compilation_combinations(self):
        for source_compiled in (False, True):
            for destination_compiled in (False, True):
                with self.subTest(source=source_compiled, destination=destination_compiled):
                    self.round_trip(source_compiled, destination_compiled)

    def test_legacy_compiled_checkpoints_work_with_both_destinations(self):
        for destination_compiled in (False, True):
            with self.subTest(destination=destination_compiled):
                self.round_trip(True, destination_compiled, legacy=True)

    def test_incompatible_architecture_still_fails_strictly(self):
        path = self.root / 'incompatible.pth'
        torch.save({'model_state_dict': self.model().state_dict()}, path)
        checkpoint = load_checkpoint(path)
        with self.assertRaises(RuntimeError):
            torch.nn.Linear(3, 2).load_state_dict(checkpoint['model_state_dict'])


if __name__ == '__main__':
    unittest.main()
