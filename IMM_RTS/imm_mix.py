# Same IMM except the mixing preserves accel state (assumes CA-CV filter) of 9D

# imm_filterpy.py
from __future__ import annotations

import numpy as np
from utils import gaussian_pdf
from filterpy.kalman import KalmanFilter
from scipy.linalg import fractional_matrix_power


class IMM:
    """
    Interacting Multiple Model filter using FilterPy
    with variable-dt transition probability matrix (TPM).
    """

    def __init__(
        self,
        filters: list[KalmanFilter],
        PI: np.ndarray,
        mu0: np.ndarray,
        t0: float = 0.0,
    ):
        self.time = float(t0)
        self.filters = filters
        self.M = len(filters)
        self.n = 9

        # Store BASE (unit-time) TPM
        self.PI_base = np.asarray(PI, dtype=float)
        self.PI = self.PI_base

        if self.PI_base.shape != (self.M, self.M):
            raise ValueError("PI must be MxM")

        mu0 = np.asarray(mu0, dtype=float)
        mu0 /= mu0.sum()
        self.mu = mu0

        self.common_dim = min(f.dim_x for f in filters)
        self.cache = []

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

        for f in self.filters:
            f.x, f.P = f.to_internal(x, P)

        self.mu = mu / np.sum(mu)

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
        self.PI = self._scale_tpm(dt)

        # --- Update model dynamics w.r.t. dt ---
        for f in self.filters:
            if hasattr(f, "dt"):
                f.dt = dt
            f.R = meas.R

        # --- Pre Mixing ---

        c_j = self.PI.T @ self.mu
        mu_ij = (self.PI * self.mu[:, None]) / c_j[None, :]
        
        x_mix = []
        P_mix = []

        # Extract Shared Statespace
        x_f = [f.x.copy() for f in self.filters]
        P_f = [f.P.copy() for f in self.filters]
        x_f_shared = x_f.copy()
        x_f_shared[0] = x_f[0][0:6]
        P_f_shared = P_f.copy()
        P_f_shared[0] = P_f[0][0:6, 0:6]
        # Mixed Shared Statespace Only
        for j in range(self.M):
            xbar = sum(mu_ij[i, j] * x_f_shared[i] for i in range(self.M))
            Pbar = sum(
                mu_ij[i, j] * (
                    P_f_shared[i] + np.outer(x_f_shared[i] - xbar, x_f_shared[i] - xbar)
                )
                for i in range(self.M)
            )
            x_mix.append(xbar)
            P_mix.append(Pbar)
        # Re-insert Augmented Components
        x_mix[0] = np.concatenate([x_mix[0], x_f[0][6:9]])
        P_mix[0] = np.block([
            [P_mix[0], P_f[0][0:6, 6:9]],
            [P_f[0][6:9,:]] ])

        # x_mix = []
        # P_mix = []
        # for j in range(self.M):
        #     x_j_ext, P_j_ext = self.filters[j].to_external(self.filters[j].x, self.filters[j].P)
        #     xbar = np.zeros(x_j_ext.shape[0])
        #     Pbar = np.zeros((x_j_ext.shape[0], x_j_ext.shape[0]))    
        #     for i in range(self.M):
        #         x_i_ext, P_i_ext = self.filters[i].to_external(self.filters[i].x, self.filters[i].P)
        #         xbar += mu_ij[i, j] * x_i_ext
        #         Pbar += mu_ij[i, j] * (
        #             P_i_ext
        #             + np.outer(x_i_ext - xbar, x_i_ext - xbar)
        #         )
        #     x_mix.append(xbar)
        #     P_mix.append(Pbar)
            

        # --- Prediction + Update ---
        
        # in internal frame
        
        x_pred, P_pred = [], []
        x_filter, P_filter = [], []
        likelihood = np.zeros(self.M)

        for j in range(self.M):
            # Predict
            self.filters[j].predict()
            x_pred.append(self.filters[j].x)
            P_pred.append(self.filters[j].P)

            # Update
            self.filters[j].update(meas.z)
            x_filter.append(self.filters[j].x)
            P_filter.append(self.filters[j].P)

            # Likelihood calc
            likelihood[j] = gaussian_pdf(meas.z, x_pred[j][0:3], P_pred[j][0:3,0:3] + meas.R)

        self.mu = c_j * likelihood
        self.mu /= np.sum(self.mu)
        
        # RTS cross-covariance  
            # This is shape of internal
        C = [P_mix[i][0:self.filters[i].dim_x, 0:self.filters[i].dim_x] @ self.filters[i].F.T for i in range(self.M)]

        # --- IMM Fusion (Combination Step) ---
        # Fused state: weighted sum of mode estimates
        
        # Need to do fusion in external
        
        x_fuse = np.zeros(self.n)
        for j in range(self.M):
            x_j, P_j = self.filters[j].to_external(self.filters[j].x, self.filters[j].P)
            x_fuse += self.mu[j] * x_j
            
        P_fuse = np.zeros((self.n, self.n))
        for j in range(self.M):
            x_j, P_j = self.filters[j].to_external(self.filters[j].x, self.filters[j].P)
            P_fuse += self.mu[j] * (P_j + np.outer(x_j - x_fuse, x_j - x_fuse))

        snapshot = {
            "x_premixed": x_mix,
            "P_premixed": P_mix,
            "F": [self.filters[i].F for i in range(self.M)],
            "x_pred": x_pred,
            "P_pred": P_pred,
            "x_filter": x_filter,
            "P_filter": P_filter,
            "x_fuse": x_fuse,
            "P_fuse": P_fuse,
            "mu": self.mu,
            "C": C,
            "PI": self.PI,
        }

        self.cache.append(snapshot)