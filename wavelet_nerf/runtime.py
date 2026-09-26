"""Execution options shared by training and evaluation."""

import argparse

import torch


def add_runtime_arguments(parser, *, training=False):
    parser.add_argument('--device', help='auto (default), cpu, cuda or cuda:<index>')
    parser.add_argument('--compile', dest='compile_model', action=argparse.BooleanOptionalAction,
                        default=None, help='Opt into torch.compile; --no-compile disables it')
    parser.add_argument('--netchunk', type=int, help='Maximum points per network query')
    if training:
        parser.add_argument('--num-workers', type=int, help='DataLoader workers (default: 0)')


def runtime_overrides(args):
    return {key: getattr(args, key) for key in ('device', 'compile_model', 'num_workers', 'netchunk')
            if getattr(args, key, None) is not None}


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' or name.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable; use --device cpu or install a working CUDA build/driver')
        index = int(name.split(':')[1]) if ':' in name else None
        if index is not None and (index < 0 or index >= torch.cuda.device_count()):
            raise ValueError(f'CUDA device index {index} is unavailable')
        device = torch.device('cuda', index)
        if index is not None:
            torch.cuda.set_device(device)
        return device
    if name != 'cpu':
        raise ValueError('Only CPU and CUDA devices are supported')
    return torch.device('cpu')


def prepare_model(model, config, device):
    if config['compile_model']:
        mode = 'reduce-overhead' if device.type == 'cuda' else 'default'
        return torch.compile(model, backend='inductor', mode=mode)
    return model
