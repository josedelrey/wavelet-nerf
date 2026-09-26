import torch


def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    """
    Apply positional encoding to the input tensor (as in NeRF).

    For each input dimension, this function appends sine and cosine functions 
    at exponentially increasing frequencies.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D).
        L (int): Number of frequency bands. For each frequency, both
            sine and cosine terms are added.

    Returns:
        torch.Tensor: Encoded tensor of shape (N, (2L + 1) * D), where the
        features are [x, sin(2^0 * x), cos(2^0 * x), ..., sin(2^{L-1} * x), cos(2^{L-1} * x)].
    """
    out = [x]
    for j in range(L):
        out.append(torch.sin(2 ** j * x))
        out.append(torch.cos(2 ** j * x))
        
    return torch.cat(out, dim=1)
