# cv.py
import numpy as np
from kalman_filter import KalmanFilterModel


class CVFilter(KalmanFilterModel):
    """
    Constant Velocity KF
    State: [x y z vx vy vz]
    Acceleration-driven process noise
    """

    def __init__(self, dt, sigma_acc, R, x0, P0, embed_sigma=50.0):
        # Store nominal dt for convenience, but prediction will accept variable dt.
        self.dt = float(dt)
        self.sigma_acc = sigma_acc
        self.R = R
        self.embed_sigma = embed_sigma

        self.x = x0.copy()
        self.P = P0.copy()

        self.x_pred = x0.copy()
        self.P_pred = P0.copy()

        # Initialize model matrices with nominal dt
        self.F = self._build_F(self.dt)
        self.H = self._build_H()
        self.Q = self._build_Q(self.dt, sigma_acc)

    @staticmethod
    def _build_F(dt):
        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)
        return F

    @staticmethod
    def _build_Q(dt, sigma):
        q = sigma**2
        Q = np.zeros((6, 6))
        for i in range(3):
            Qb = q * np.array([
                [dt**4/4, dt**3/2],
                [dt**3/2, dt**2]
            ])
            idx = [i, i+3]
            for r in range(2):
                for c in range(2):
                    Q[idx[r], idx[c]] = Qb[r, c]
        return Q

    @staticmethod
    def _build_H():
        H = np.zeros((3, 6))
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
        self.Q = self._build_Q(tau, self.sigma_acc)

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
        self.P = (np.eye(6) - K @ self.H) @ self.P_pred

    def get_state(self):
        return self.x.copy(), self.P.copy()

    def get_pred(self):
        return self.x_pred.copy(), self.P_pred.copy()

    def set_state(self, x, P):
        self.x = x[0:6].copy().reshape(6,1)
        self.P = P[0:6,0:6].copy().reshape(6,6)

    def to_common(self, x_internal: np.ndarray, P_internal: np.ndarray):
        x = np.zeros((9, 1))
        P = np.zeros((9, 9))
        x[0:6] = x_internal
        P[0:6, 0:6] = P_internal
        P[6:9, 6:9] = self.embed_sigma**2 * np.eye(3)
        return x, P

    def to_internal(self, x_common, P_common):
        return x_common[0:6].copy(), P_common[0:6, 0:6].copy()

    @property
    def dim(self):
        return 6

    @property
    def common_dim(self):
        return 9
