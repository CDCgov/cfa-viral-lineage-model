import numpy as np
from numpy.typing import ArrayLike


def crps(samples: ArrayLike, truth: float):
    """
    Monte Carlo approximation to the CRPS.
    """

    return (
        np.abs(samples - truth).mean()
        - 0.5 * np.abs(samples[:-1] - samples[1:]).mean()
    )
