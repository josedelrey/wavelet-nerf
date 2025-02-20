import torch


def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    """
    Applies positional encoding to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        L (int): Number of encoding functions (frequencies).

    Returns:
        torch.Tensor: Concatenated tensor with positional encodings.
    """
    out = [x]
    for j in range(L):
        out.append(torch.sin(2 ** j * x))
        out.append(torch.cos(2 ** j * x))
    return torch.cat(out, dim=1)
