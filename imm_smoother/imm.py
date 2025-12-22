# imm.py
from __future__ import annotations

import numpy as np


def _gaussian_likelihood(y: np.ndarray, S: np.ndarray) -> float:
    """
    Compute N(y; 0, S) likelihood (scalar).
    y: innovation (m,)
    S: innovation covariance (m,m)
    """
    m = y.shape[0]
    # Numerical stability
    try:
        chol = np.linalg.cholesky(S)
        # Solve for mahalanobis: y^T S^{-1} y
        sol = np.linalg.solve(chol, y)
        maha = float(sol.T @ sol)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    except np.linalg.LinAlgError:
        # Fallback
        S_inv = np.linalg.pinv(S)
        maha = float(y.T @ S_inv @ y)
        sign, logdet = np.linalg.slogdet(S + 1e-12 * np.eye(m))
        log_det = float(logdet) if sign > 0 else float(np.log(np.maximum(np.linalg.det(S), 1e-300)))

    log_norm = -0.5 * (m * np.log(2.0 * np.pi) + log_det)
    log_exp = -0.5 * maha
    return float(np.exp(log_norm + log_exp))


class LinearKalmanFilter:
    """
    Standard discrete-time linear Kalman Filter:
        x_k = F x_{k-1} + w,   w ~ N(0,Q)
        z_k = H x_k + v,       v ~ N(0,R)
    """
    def __init__(
        self,
        F: np.ndarray,
        Q: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        self.F = np.asarray(F, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (innovation y, innovation covariance S, kalman gain K)
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ self.H.T @ S_inv
        I = np.eye(self.P.shape[0])

        self.x = self.x + K @ y
        self.P = (I - K @ self.H) @ self.P

        return y, S, K

    def innovation_stats(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return y, S


class IMM:
    """
    Interacting Multiple Model (IMM) filter supporting a CA (9-state) and CV (6-state) model.

    Models:
      - "CA": state = [x y z vx vy vz ax ay az]^T (9x1)
      - "CV": state = [x y z vx vy vz]^T         (6x1)

    Measurements:
      - z = [x y z]^T (3x1)

    Mixing is done in a common 9D space by embedding CV states with ax,ay,az=0.
    """

    COMMON_DIM = 9

    def __init__(
        self,
        dt: float,
        R: np.ndarray,
        # Process noise parameters
        sigma_acc_ca: float,
        sigma_acc_cv: float,
        # Model switching probabilities matrix PI (NxN). PI[i,j]=P(model_j | model_i)
        PI: np.ndarray,
        # Initial model probabilities (N,)
        mu0: np.ndarray,
        # Initial states/covariances for each model
        x0_ca: np.ndarray,
        P0_ca: np.ndarray,
        x0_cv: np.ndarray,
        P0_cv: np.ndarray,
        # Embedding variance for missing accel when lifting CV->common during mixing
        embed_accel_sigma: float = 50.0,
        # Time-weighted likelihood
        time_weighted_likelihood: bool = False,
        likelihood_tau: float = 1.0,
    ):
        self.dt = float(dt)
        self.R = np.asarray(R, dtype=float)
        self.PI = np.asarray(PI, dtype=float)
        self.mu = np.asarray(mu0, dtype=float).reshape(-1)

        if self.PI.shape != (2, 2):
            raise ValueError("This IMM implementation expects exactly 2 models: CA and CV.")
        if self.mu.shape != (2,):
            raise ValueError("mu0 must be shape (2,) corresponding to [CA, CV].")

        self.embed_accel_sigma = float(embed_accel_sigma)

        # Build model filters
        # CA model
        F_ca = self._F_ca(self.dt)
        Q_ca = self._Q_ca(self.dt, sigma_acc_ca)
        H_ca = self._H_ca(dim=9)
        self.kf_ca = LinearKalmanFilter(F_ca, Q_ca, H_ca, self.R, x0_ca, P0_ca)

        # CV model
        F_cv = self._F_cv(self.dt)
        Q_cv = self._Q_cv(self.dt, sigma_acc_cv)
        H_cv = self._H_ca(dim=6)  # same H form but smaller state
        self.kf_cv = LinearKalmanFilter(F_cv, Q_cv, H_cv, self.R, x0_cv, P0_cv)

        self.time_weighted_likelihood = bool(time_weighted_likelihood)
        self.likelihood_tau = float(likelihood_tau)

    # ---------------------- Model matrices ----------------------

    @staticmethod
    def _F_cv(dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    @staticmethod
    def _Q_cv(dt: float, sigma_acc: float) -> np.ndarray:
        """
        CV process noise via white acceleration model per axis:
        x, v driven by accel noise with PSD ~ sigma_acc^2.
        For each axis, Q_block = sigma^2 * [[dt^4/4, dt^3/2],
                                            [dt^3/2, dt^2]]
        """
        q = float(sigma_acc) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q2 = q * np.array([[dt4 / 4.0, dt3 / 2.0],
                           [dt3 / 2.0, dt2]], dtype=float)
        Q = np.zeros((6, 6), dtype=float)
        for axis in range(3):
            i = axis
            v = axis + 3
            Q[i, i] = Q2[0, 0]
            Q[i, v] = Q2[0, 1]
            Q[v, i] = Q2[1, 0]
            Q[v, v] = Q2[1, 1]
        return Q

    @staticmethod
    def _F_ca(dt: float) -> np.ndarray:
        """
        CA with constant acceleration:
        p += v*dt + 0.5*a*dt^2
        v += a*dt
        a += a (constant)
        """
        F = np.eye(9)
        # position from velocity
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # position from acceleration
        half_dt2 = 0.5 * dt * dt
        F[0, 6] = half_dt2
        F[1, 7] = half_dt2
        F[2, 8] = half_dt2
        # velocity from acceleration
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt
        return F

    @staticmethod
    def _Q_ca(dt: float, sigma_jerk: float) -> np.ndarray:
        """
        CA process noise via white jerk model per axis.
        For 1D state [p v a], with jerk noise variance sigma^2:
        Q = sigma^2 * [[dt^5/20, dt^4/8,  dt^3/6],
                       [dt^4/8,  dt^3/3,  dt^2/2],
                       [dt^3/6,  dt^2/2,  dt]]
        """
        q = float(sigma_jerk) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        dt5 = dt4 * dt
        Q3 = q * np.array([[dt5 / 20.0, dt4 / 8.0,  dt3 / 6.0],
                           [dt4 / 8.0,  dt3 / 3.0,  dt2 / 2.0],
                           [dt3 / 6.0,  dt2 / 2.0,  dt]], dtype=float)
        Q = np.zeros((9, 9), dtype=float)
        for axis in range(3):
            p = axis
            v = axis + 3
            a = axis + 6
            idx = [p, v, a]
            for r in range(3):
                for c in range(3):
                    Q[idx[r], idx[c]] = Q3[r, c]
        return Q

    @staticmethod
    def _H_ca(dim: int) -> np.ndarray:
        """
        Position-only measurement: z = [x y z]
        """
        H = np.zeros((3, dim), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    # ---------------------- Embedding / projection ----------------------

    def _to_common(self, model: str, x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        P = np.asarray(P, dtype=float)
        if model == "CA":
            if x.shape[0] != 9:
                raise ValueError("CA state must be 9x1")
            return x.copy(), P.copy()

        if model == "CV":
            if x.shape[0] != 6:
                raise ValueError("CV state must be 6x1")
            xc = np.zeros((9, 1), dtype=float)
            xc[0:3, 0] = x[0:3, 0]
            xc[3:6, 0] = x[3:6, 0]
            # accel assumed 0
            Pc = np.zeros((9, 9), dtype=float)
            Pc[0:6, 0:6] = P
            # large uncertainty in embedded accel
            s2 = self.embed_accel_sigma ** 2
            Pc[6, 6] = s2
            Pc[7, 7] = s2
            Pc[8, 8] = s2
            return xc, Pc

        raise ValueError(f"Unknown model '{model}'")

    @staticmethod
    def _from_common_to_ca(xc: np.ndarray, Pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return xc.copy(), Pc.copy()

    @staticmethod
    def _from_common_to_cv(xc: np.ndarray, Pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = xc[0:6, :].copy()
        P = Pc[0:6, 0:6].copy()
        return x, P

    # ---------------------- IMM step ----------------------

    def step(self, z: np.ndarray) -> dict:
        """
        Perform one IMM update with measurement z (3,).

        Returns dict with:
          - x_combined_common (9,1)
          - P_combined_common (9,9)
          - mu (2,) updated model probabilities [CA, CV]
          - x_ca, P_ca, x_cv, P_cv
        """
        # Current model states
        x_ca, P_ca = self.kf_ca.x, self.kf_ca.P
        x_cv, P_cv = self.kf_cv.x, self.kf_cv.P

        # Mixing probabilities
        # c_j = sum_i mu_i * PI[i,j]
        c = self.mu @ self.PI  # (2,)
        c = np.maximum(c, 1e-12)
        # mu_ij = (mu_i * PI[i,j]) / c_j
        mu_ij = (self.mu.reshape(-1, 1) * self.PI) / c.reshape(1, -1)  # (2,2)

        # Convert each model estimate to common (9D)
        x_ca_c, P_ca_c = self._to_common("CA", x_ca, P_ca)
        x_cv_c, P_cv_c = self._to_common("CV", x_cv, P_cv)

        Xc = [x_ca_c, x_cv_c]
        Pc = [P_ca_c, P_cv_c]

        # Mixed initial conditions in common space for each destination model j
        mixed_common = []
        for j in range(2):
            # x0_j = sum_i mu_ij[i,j] * x_i
            x0 = mu_ij[0, j] * Xc[0] + mu_ij[1, j] * Xc[1]
            # P0_j = sum_i mu_ij[i,j] * (P_i + (x_i - x0)(x_i - x0)^T)
            P0 = np.zeros((9, 9), dtype=float)
            for i in range(2):
                dx = Xc[i] - x0
                P0 += mu_ij[i, j] * (Pc[i] + dx @ dx.T)
            mixed_common.append((x0, P0))

        # Set mixed initial to each filter (convert to their dims)
        x0_ca, P0_ca = self._from_common_to_ca(*mixed_common[0])
        x0_cv, P0_cv = self._from_common_to_cv(*mixed_common[1])

        self.kf_ca.x, self.kf_ca.P = x0_ca, P0_ca
        self.kf_cv.x, self.kf_cv.P = x0_cv, P0_cv

        # Predict + update each model; compute likelihood
        # CA
        self.kf_ca.predict()
        y_ca, S_ca = self.kf_ca.innovation_stats(z)
        self.kf_ca.update(z)
        L_ca = _gaussian_likelihood(y_ca, S_ca)

        # CV
        self.kf_cv.predict()
        y_cv, S_cv = self.kf_cv.innovation_stats(z)
        self.kf_cv.update(z)
        L_cv = _gaussian_likelihood(y_cv, S_cv)

        # Update model probabilities
        # mu_j ∝ L_j * c_j
        # ---- Model probability update (optionally time-weighted) ----
        if self.time_weighted_likelihood:
            alpha = self.dt / self.likelihood_tau
            L_ca_eff = L_ca ** alpha
            L_cv_eff = L_cv ** alpha
        else:
            L_ca_eff = L_ca
            L_cv_eff = L_cv

        mu_new = np.array([
            L_ca_eff * c[0],
            L_cv_eff * c[1],
        ], dtype=float)
        
        mu_sum = float(np.sum(mu_new))
        if mu_sum <= 0:
            mu_new = np.array([0.5, 0.5], dtype=float)
        else:
            mu_new /= mu_sum
        self.mu = mu_new

        # Combine estimates in common 9D
        x_ca_c2, P_ca_c2 = self._to_common("CA", self.kf_ca.x, self.kf_ca.P)
        x_cv_c2, P_cv_c2 = self._to_common("CV", self.kf_cv.x, self.kf_cv.P)

        x_comb = self.mu[0] * x_ca_c2 + self.mu[1] * x_cv_c2
        P_comb = (
            self.mu[0] * (P_ca_c2 + (x_ca_c2 - x_comb) @ (x_ca_c2 - x_comb).T)
            + self.mu[1] * (P_cv_c2 + (x_cv_c2 - x_comb) @ (x_cv_c2 - x_comb).T)
        )

        return {
            "x_combined_common": x_comb,
            "P_combined_common": P_comb,
            "mu": self.mu.copy(),
            "x_ca": self.kf_ca.x.copy(),
            "P_ca": self.kf_ca.P.copy(),
            "x_cv": self.kf_cv.x.copy(),
            "P_cv": self.kf_cv.P.copy(),
            "likelihoods": (L_ca, L_cv),
        }
