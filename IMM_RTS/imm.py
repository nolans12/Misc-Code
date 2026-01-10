# imm_filterpy.py
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator
from scipy.linalg import fractional_matrix_power


class IMM:
    """
    Interacting Multiple Model filter using FilterPy.
    """

    def __init__(
        self,
        models: list[KalmanFilter],
        PI: np.ndarray,
        mu0: np.ndarray,
        dt: float,
        t0: float = 0.0,
    ):
        self.time = float(t0)
        self.models = models
        self.M = len(models)
        self.dt = float(dt)

        PI = np.asarray(PI, dtype=float)

        if PI.shape != (self.M, self.M):
            raise ValueError("PI must be MxM")

        # Time-scaled TPM (optional but safe)
        PI = fractional_matrix_power(PI, dt)
        PI = np.maximum(PI, 0.0)
        PI /= PI.sum(axis=1, keepdims=True)

        mu0 = np.asarray(mu0, dtype=float)
        mu0 /= np.sum(mu0)

        self.imm = IMMEstimator(models, mu0, PI)

        self.common_dim = models[0].dim_x
        self._history = []

    def set_state(self, x, P, mu):
        x = np.asarray(x).reshape(-1)  # <-- FORCE (9,)
        P = np.asarray(P)

        for f in self.imm.filters:
            f.x = x.copy()
            f.P = P.copy()

        self.imm.mu = mu / np.sum(mu)

    def step(self, meas) -> dict:
        """
        meas must have:
            meas.z : measurement vector
            meas.R : measurement covariance
            meas.t : timestamp
        """

        dt = meas.t - self.time
        self.time = meas.t

        # Update model dynamics if needed
        for f in self.imm.filters:
            if hasattr(f, "dt"):
                f.dt = dt
            f.R = meas.R

        # IMM predict + update
        self.imm.predict()
        self.imm.update(meas.z)

        x_comb = self.imm.x.copy()
        P_comb = self.imm.P.copy()
        mu = self.imm.mu.copy()

        # Save history (optional)
        snapshot = {
            "time": self.time,
            "x_common": x_comb,
            "P_common": P_comb,
            "mu": mu.copy(),
            "p_trans": self.imm.M.copy(),
            "measurement": meas,
            "models": [],
        }

        for f in self.imm.filters:
            snapshot["models"].append({
                "x_filt": f.x.copy(),
                "P_filt": f.P.copy(),
                "x_pred": f.x_prior.copy(),
                "P_pred": f.P_prior.copy(),
                "F": f.F.copy(),
            })

        self._history.append(snapshot)

        return {
            "x_common": x_comb,
            "P_common": P_comb,
            "mu": mu,
            "likelihoods": self.imm.likelihood.copy(),
        }
