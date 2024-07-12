import numpy as np
from numpy.typing import ArrayLike


def crps(samples: ArrayLike, truth: float):
    """
    Monte Carlo approximation to the CRPS.
    """

    return (
        np.abs(samples - truth).mean()
        - 0.5 * np.abs(samples[::2] - samples[1::2]).mean()
        # TODO: 10.1007/s11749-008-0114-x uses overlapping pairs of samples
        # but I am not sure if this is safe
        # - 0.5 * np.abs(samples[:-1] - samples[1:]).mean()
    )
