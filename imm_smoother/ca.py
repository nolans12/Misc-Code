# ca.py
import numpy as np
from kalman_filter import KalmanFilterModel


class CAFilter(KalmanFilterModel):
    """
    Constant Acceleration KF
    State: [x y z vx vy vz ax ay az]
    Jerk-driven process noise
    """

    def __init__(self, dt, sigma_jerk, R, x0, P0):
        # Store nominal dt for convenience, but prediction will accept variable dt.
        self.dt = float(dt)
        self.sigma_jerk = sigma_jerk
        self.R = R

        self.x = x0.copy()
        self.P = P0.copy()

        self.x_pred = x0.copy()
        self.P_pred = P0.copy()

        # Initialize model matrices with nominal dt
        self.F = self._build_F(self.dt)
        self.H = self._build_H()
        self.Q = self._build_Q(self.dt, sigma_jerk)

    @staticmethod
    def _build_F(dt):
        F = np.eye(9)
        F[0:3, 3:6] = dt * np.eye(3)
        F[0:3, 6:9] = 0.5 * dt**2 * np.eye(3)
        F[3:6, 6:9] = dt * np.eye(3)
        return F

    @staticmethod
    def _build_Q(dt, sigma):
        q = sigma**2
        Q = np.zeros((9, 9))
        for i in range(3):
            Qb = q * np.array([
                [dt**5/20, dt**4/8,  dt**3/6],
                [dt**4/8,  dt**3/3,  dt**2/2],
                [dt**3/6,  dt**2/2,  dt]
            ])
            idx = [i, i+3, i+6]
            for r in range(3):
                for c in range(3):
                    Q[idx[r], idx[c]] = Qb[r, c]
        return Q

    @staticmethod
    def _build_H():
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        return H

    def predict(self, dt: float):
        """
        Propagate state with a variable time step dt.

        The transition F uses signed dt so that negative dt propagates backward
        in time. The process noise covariance uses |dt| so that noise power
        depends only on the magnitude of the time interval.
        """
        # Signed dt for dynamics
        self.F = self._build_F(dt)
        # |dt| for process noise magnitude
        tau = abs(dt)
        self.Q = self._build_Q(tau, self.sigma_jerk)

        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z):
        y = z.reshape(-1, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z):
        y, S = self.innovation(z)
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = self.x_pred + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P_pred

    def get_state(self):
        return self.x.copy(), self.P.copy()

    def get_pred(self):
        return self.x_pred.copy(), self.P_pred.copy()

    def set_state(self, x, P):
        self.x = x[0:9].copy().reshape(9,1)
        self.P = P[0:9,0:9].copy().reshape(9,9)

    def to_common(self, x_internal: np.ndarray, P_internal: np.ndarray):
        return x_internal.copy(), P_internal.copy()

    def to_internal(self, x_common, P_common):
        return x_common[0:9].copy(), P_common[0:9, 0:9].copy()

    @property
    def dim(self):
        return 9

    @property
    def common_dim(self):
        return 9
