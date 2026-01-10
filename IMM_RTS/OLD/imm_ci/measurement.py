from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Measurement:
    """
    Single position measurement with timestamp and covariance.

    Attributes
    ----------
    t : float
        Measurement time (seconds).
    z : np.ndarray
        Measured position vector with shape (3,) or (3, 1).
    R : np.ndarray
        Measurement covariance matrix with shape (3, 3).
    """

    t: float
    z: np.ndarray
    R: np.ndarray

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.z = np.asarray(self.z, dtype=float).reshape(3)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)


