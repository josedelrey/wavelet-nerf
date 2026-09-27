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
    values = np.asarray(mse, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("MSE must be finite and nonnegative")
    result = np.full(values.shape, np.inf)
    positive = values > 0
    result[positive] = -10 * np.log10(values[positive])
    return float(result) if result.ndim == 0 else result
