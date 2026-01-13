# imm_filterpy.py
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator
from scipy.linalg import fractional_matrix_power


class IMM:
    """
    Interacting Multiple Model filter using FilterPy
    with variable-dt transition probability matrix (TPM).
    """

    def __init__(
        self,
        models: list[KalmanFilter],
        PI: np.ndarray,
        mu0: np.ndarray,
        t0: float = 0.0,
    ):
        self.time = float(t0)
        self.models = models
        self.M = len(models)

        # Store BASE (unit-time) TPM
        self.PI_base = np.asarray(PI, dtype=float)

        if self.PI_base.shape != (self.M, self.M):
            raise ValueError("PI must be MxM")

        mu0 = np.asarray(mu0, dtype=float)
        mu0 /= mu0.sum()

        self.imm = IMMEstimator(models, mu0, self.PI_base)

        self.common_dim = models[0].dim_x
        self._history = []

    # ------------------------------------------------------------------
    # TPM scaling
    # ------------------------------------------------------------------
    def _scale_tpm(self, dt: float) -> np.ndarray:
        """
        Scale the base TPM according to dt using matrix fractional power.
        Ensures numerical safety and row normalization.
        """
        PI_dt = fractional_matrix_power(self.PI_base, dt)

        # Numerical cleanup
        PI_dt = np.real_if_close(PI_dt)
        PI_dt = np.maximum(PI_dt, 0.0)

        row_sums = PI_dt.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        PI_dt /= row_sums

        return PI_dt

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------
    def set_state(self, x, P, mu):
        x = np.asarray(x).reshape(-1)
        P = np.asarray(P)

        for f in self.imm.filters:
            f.x, f.P = x.copy(), P.copy()

        self.imm.mu = mu / np.sum(mu)

    # ------------------------------------------------------------------
    # Predict + update step
    # ------------------------------------------------------------------
    def step(self, meas) -> dict:
        """
        meas must have:
            meas.z : measurement vector
            meas.R : measurement covariance
            meas.t : timestamp
        """

        dt = meas.t - self.time
        self.time = meas.t

        # --- Update TPM for this dt ---
        self.imm.PI = self._scale_tpm(dt)

        # --- Update model dynamics ---
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
            "PI_dt": self.imm.PI.copy(),
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
