# imm.py
from __future__ import annotations

import numpy as np
from kalman_filter import KalmanFilterModel
from measurement import Measurement


def gaussian_likelihood(y: np.ndarray, S: np.ndarray) -> float:
    m = y.shape[0]
    try:
        chol = np.linalg.cholesky(S)
        sol = np.linalg.solve(chol, y)
        maha = sol.T @ sol
        log_det = 2 * np.sum(np.log(np.diag(chol)))
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        maha = y.T @ S_inv @ y
        sign, log_det = np.linalg.slogdet(S)
        log_det = log_det if sign > 0 else np.log(np.maximum(np.linalg.det(S), 1e-300))

    logL = -0.5 * (m * np.log(2 * np.pi) + log_det + maha)
    return float(np.exp(logL))


class IMM:
    """
    Generic Interacting Multiple Model (IMM) filter.
    """

    def __init__(
        self,
        models: list[KalmanFilterModel],
        PI: np.ndarray,
        mu0: np.ndarray,
        dt: float,
        time_weighted_likelihood: bool = False,
        likelihood_tau: float = 1.0,
        t0: float = 0.0,
    ):
        """
        Parameters
        ----------
        models : list[KalmanFilterModel]
            Underlying KF models (e.g. CA, CV).
        PI : np.ndarray
            Mode transition matrix, shape (M, M).
        mu0 : np.ndarray
            Initial mode probabilities, shape (M,).
        dt : float
            Nominal time step, used only for default time-weighted likelihood.
        time_weighted_likelihood : bool
            If True, apply exponential time-weighting to likelihoods.
        likelihood_tau : float
            Time constant for likelihood weighting.
        t0 : float
            Initial time of the filter state.
        """
        self.time = float(t0)
        self.models = models
        self.M = len(models)
        self.PI = np.asarray(PI, dtype=float)
        self.mu = np.asarray(mu0, dtype=float)
        self.dt = float(dt)
        self._history = []  # stores forward pass snapshots

        self.time_weighted_likelihood = time_weighted_likelihood
        self.likelihood_tau = likelihood_tau

        if self.PI.shape != (self.M, self.M):
            raise ValueError("PI must be MxM")

        if self.mu.shape != (self.M,):
            raise ValueError("mu0 must have length M")

        self.common_dim = models[0].common_dim
        for m in models:
            if m.common_dim != self.common_dim:
                raise ValueError("All models must share common_dim")

    def set_state(self, est, cov, mu):
        for m in self.models:
            m.set_state(est, cov)
        self.mu = mu

    def step(self, meas: Measurement) -> dict:
        """
        Advance the IMM with a new measurement.

        The time step dt is inferred from the measurement timestamp, so
        feeding measurements in reverse time order automatically uses
        negative dt for prediction.
        """
        # Compute variable (possibly negative) dt from measurement time.
        dt = meas.t - self.time
        self.time = meas.t

        z = meas.z

        # ---- Mixing probabilities ----
        c = self.mu @ self.PI
        c = np.maximum(c, 1e-12)

        mu_ij = (self.mu[:, None] * self.PI) / c[None, :]

        # ---- Convert all models to common space ----
        Xc, Pc = [], []
        for model in self.models:
            x, P = model.to_common(model.x, model.P)
            Xc.append(x)
            Pc.append(P)

        # ---- Mixing ----
        mixed_common = []
        for j in range(self.M):
            x0 = sum(mu_ij[i, j] * Xc[i] for i in range(self.M))
            P0 = np.zeros((self.common_dim, self.common_dim))
            for i in range(self.M):
                dx = Xc[i] - x0
                P0 += mu_ij[i, j] * (Pc[i] + dx @ dx.T)
            mixed_common.append((x0, P0))

        # ---- Set mixed states ----
        for j, model in enumerate(self.models):
            x_c, P_c = model.to_internal(*mixed_common[j])
            model.set_state(x_c, P_c)

        # ---- Predict, update, likelihood ----
        likelihoods = np.zeros(self.M)
        for j, model in enumerate(self.models):
            # Use measurement covariance from this sample
            if hasattr(model, "R"):
                model.R = meas.R

            # print(f"Predicting model {j} at time {self.time} with dt {dt}")
            # Variable-dt prediction (dt can be negative for backward pass)
            model.predict(dt)
            y, S = model.innovation(z)
            model.update(z)
            likelihoods[j] = gaussian_likelihood(y, S)

        # ---- Update model probabilities ----
        if self.time_weighted_likelihood:
            # Use |dt| so weighting is independent of direction of time.
            alpha = abs(dt) / self.likelihood_tau
            likelihoods = likelihoods ** alpha

        mu_new = likelihoods * c
        mu_sum = np.sum(mu_new)
        self.mu = mu_new / mu_sum if mu_sum > 0 else np.ones(self.M) / self.M

        # ---- Combine estimate ----
        Xc_new, Pc_new = [], []
        for model in self.models:
            x, P = model.to_common(model.x, model.P)
            Xc_new.append(x)
            Pc_new.append(P)

        x_comb = sum(self.mu[j] * Xc_new[j] for j in range(self.M))
        P_comb = np.zeros((self.common_dim, self.common_dim))
        for j in range(self.M):
            dx = Xc_new[j] - x_comb
            P_comb += self.mu[j] * (Pc_new[j] + dx @ dx.T)

        # ---- Save history for smoothing ----
        snapshot = {
            "time": self.time,
            "models": [],
            "mu": self.mu.copy(),
        }
        for model in self.models:
            x_filt, P_filt = model.get_state()
            x_pred, P_pred = model.get_pred()
            snapshot["models"].append({
                "x_filt": x_filt.copy(),
                "P_filt": P_filt.copy(),
                "x_pred": x_pred.copy(),
                "P_pred": P_pred.copy(),
                "F": model.F.copy(),  # required for RTS
            })
        self._history.append(snapshot)

        return {
            "x_common": x_comb,
            "P_common": P_comb,
            "mu": self.mu.copy(),
            "likelihoods": likelihoods.copy(),
        }

