# kalman_filter.py
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class KalmanFilterModel(ABC):
    """
    Abstract base class for IMM-compatible Kalman filters.
    """

    @abstractmethod
    def predict(self, dt: float) -> None:
        """
        Propagate the state and covariance by a time step dt.

        dt can be positive (forward in time) or negative (backward in time).
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        pass

    @abstractmethod
    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (innovation y, innovation covariance S)
        """
        pass

    @abstractmethod
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (x, P)
        """
        pass

    @abstractmethod
    def get_pred(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (x pred, P pred)
        """
        pass

    @abstractmethod
    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        pass

    @abstractmethod
    def to_common(self, x_internal: np.ndarray, P_internal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert model state to common IMM state space.
        """
        pass

    @abstractmethod
    def to_internal(self, x_common: np.ndarray, P_common: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize model state from common IMM state.
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def common_dim(self) -> int:
        pass
