import numpy as np
from filterpy.kalman import KalmanFilter


def make_ca_3d(dt: float, sigma: float, R: np.ndarray) -> KalmanFilter:
    f = KalmanFilter(dim_x=9, dim_z=3)

    f.x = np.zeros(9)

    dt2 = 0.5 * dt**2

    f.F = np.array([
        [1, 0, 0, dt, 0,  0,  dt2, 0,   0],
        [0, 1, 0, 0,  dt, 0,  0,   dt2, 0],
        [0, 0, 1, 0,  0,  dt, 0,   0,   dt2],

        [0, 0, 0, 1,  0,  0,  dt,  0,   0],
        [0, 0, 0, 0,  1,  0,  0,   dt,  0],
        [0, 0, 0, 0,  0,  1,  0,   0,   dt],

        [0, 0, 0, 0,  0,  0,  1,   0,   0],
        [0, 0, 0, 0,  0,  0,  0,   1,   0],
        [0, 0, 0, 0,  0,  0,  0,   0,   1],
    ])

    f.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
    ])

    f.Q = _build_Q_ca(dt, sigma)

    f.R = R
    f.P = np.eye(9) * 10.0

    return f

def to_external_ca(self, x_internal: np.ndarray, P_internal: np.ndarray):
    return x_internal.copy(), P_internal.copy()

def to_internal_ca(self, x_common, P_common):
    return x_common[0:9].copy(), P_common[0:9, 0:9].copy()

def make_cv_3d(dt: float, sigma: float, R: np.ndarray) -> KalmanFilter:
    f = KalmanFilter(dim_x=9, dim_z=3)

    f.x = np.zeros(9)

    # State transition (CV embedded in CA)
    f.F = np.array([
        [1, 0, 0, dt, 0,  0,  0,  0,  0],
        [0, 1, 0, 0,  dt, 0,  0,  0,  0],
        [0, 0, 1, 0,  0,  dt, 0,  0,  0],

        [0, 0, 0, 1,  0,  0,  0,  0,  0],
        [0, 0, 0, 0,  1,  0,  0,  0,  0],
        [0, 0, 0, 0,  0,  1,  0,  0,  0],

        [0, 0, 0, 0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0,  0,  0,  0,  0,  0],
    ])

    # Measure position only
    f.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
    ])

    f.Q = _build_Q_cv(dt, sigma)

    f.R = R
    f.P = np.eye(9) * 10.0

    return f

def to_external_cv(self, x_internal: np.ndarray, P_internal: np.ndarray):
    x = np.zeros((9))
    P = np.zeros((9, 9))
    x[0:6] = x_internal[0:6]
    P[0:6, 0:6] = P_internal[0:6,0:6]
    return x, P

def to_internal_cv(self, x_common, P_common):
    return x_common[0:6].copy(), P_common[0:6, 0:6].copy()

def _build_Q_cv(dt, sigma):
    q = sigma ** 2
    Q = np.zeros((9, 9))
    for i in range(3):
        Qb = q * np.array([
            [dt ** 4 / 4, dt ** 3 / 2],
            [dt ** 3 / 2, dt ** 2]
        ])
        idx = [i, i + 3]
        for r in range(2):
            for c in range(2):
                Q[idx[r], idx[c]] = Qb[r, c]
    return Q

def _build_Q_ca(dt, sigma):
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