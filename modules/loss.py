import numpy as np


def mse_to_psnr(mse: float) -> float:
    """
    Convert Mean Squared Error (MSE) to Peak Signal-to-Noise Ratio (PSNR).

    The formula used is:
        PSNR = 20 * log10(MAX_I / sqrt(MSE))
    where MAX_I = 1.0 (assuming input images are normalized to [0, 1]).

    Args:
        mse (float or np.ndarray): Mean Squared Error value(s).

    Returns:
        float or np.ndarray: PSNR value(s). Returns +inf if mse = 0.
    """
    return 20 * np.log10(1 / np.sqrt(mse))
