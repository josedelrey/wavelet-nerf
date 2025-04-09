import numpy as np


def mse_to_psnr(mse: float) -> float:
    """
    Converts Mean Squared Error to Peak Signal-to-Noise Ratio.

    Args:
        mse (float): Mean Squared Error.

    Returns:
        float: PSNR value.
    """
    return 20 * np.log10(1 / np.sqrt(mse))
