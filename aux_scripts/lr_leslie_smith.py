import os, sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

from modules.data import load_dataset, compute_rays, RayDataset
from modules.models import NeRF, Siren, WaveletNeRF
from modules.rendering import render_nerf


def main():
    parser = argparse.ArgumentParser(
        description="LR range test for NeRF variants with TensorBoard"
    )
    parser.add_argument('--model',     choices=['nerf','siren','wavelet'],
                        default='wavelet')
    parser.add_argument('--lr_start',  type=float, default=1e-6,
                        help='Starting LR for range test')
    parser.add_argument('--lr_end',    type=float, default=1e-1,
                        help='Ending LR for range test')
    parser.add_argument('--num_iters', type=int,   default=5000,
                        help='Number of iterations over which to sweep')
    parser.add_argument('--logdir',    type=str,   default='./logs/lr_find_leslie_smith',
                        help='TensorBoard log directory')
    args = parser.parse_args()

    # Prepare logdir
    writer = SummaryWriter(log_dir=args.logdir)

    # Load data
    images_np, c2w_np, focal = load_dataset('./datasets/lego', mode='train')
    rays_o, rays_d, target = compute_rays(images_np, c2w_np, focal)
    ds = RayDataset(rays_o, rays_d, target)
    loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)
    it = iter(loader)

    # Select model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'nerf':
        model = NeRF().to(device)
    elif args.model == 'siren':
        model = Siren().to(device)
    else:
        model = WaveletNeRF().to(device)

    # Optimizer (lr overwritten each step)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    loss_fn   = nn.MSELoss()

    lrs = []
    losses = []
    # LR sweep loop
    for step in tqdm(range(args.num_iters), desc="LR range test", unit="it"):  
        try:
            rays_o_b, rays_d_b, tgt_b = next(it)
        except StopIteration:
            it = iter(loader)
            rays_o_b, rays_d_b, tgt_b = next(it)

        # Compute LR
        frac = step / (args.num_iters - 1)
        lr   = args.lr_start * (args.lr_end/args.lr_start) ** frac
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward + backward
        rays_o_b = rays_o_b.to(device)
        rays_d_b = rays_d_b.to(device)
        tgt_b    = tgt_b.to(device)

        pred = render_nerf(
            model, rays_o_b, rays_d_b,
            near=2.0, far=6.0,
            num_samples=128,
            device=device,
            white_background=True,
            chunk_size=1024
        )

        optimizer.zero_grad()
        loss = loss_fn(pred, tgt_b)
        loss.backward()
        optimizer.step()

        # Log to TensorBoard
        writer.add_scalar('lr_finder/lr',   lr,   step)
        writer.add_scalar('lr_finder/loss', loss.item(), step)

        lrs.append(lr)
        losses.append(loss.item())

        if step % 100 == 0:
            print(f"[{step:4d}/{args.num_iters}]  lr={lr:.3e}  loss={loss.item():.6f}")
    
    # Plot LR vs Loss
    fig, ax = plt.subplots()
    ax.loglog(lrs, losses, marker='.')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('LR Range Test')
    
    writer.add_figure('lr_range_test/plot', fig)
    plt.close(fig)
    
    writer.close()
    print(f"LR range test complete. Visualize with: tensorboard --logdir={args.logdir}")

    best_idx = int(torch.tensor(losses).argmin().item())
    best_lr  = lrs[best_idx]
    print(f"Optimal LR ≃ {best_lr:.3e} at step {best_idx}")


if __name__ == '__main__':
    main()
