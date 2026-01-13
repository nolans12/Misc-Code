import numpy as np

def cv_model_2d(dt, sigma_a2):
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Q = np.diag([0, 0, 2*sigma_a2*dt, 2*sigma_a2*dt])
    return F, Q

def ca_model_2d(dt, sigma_j2):
    F = np.array([
        [1, 0, dt,  0, 0.5*dt**2,         0],
        [0, 1,  0, dt,         0, 0.5*dt**2],
        [0, 0,  1,  0,        dt,         0],
        [0, 0,  0,  1,         0,        dt],
        [0, 0,  0,  0,         1,         0],
        [0, 0,  0,  0,         0,         1]
    ])
    Q = np.diag([0, 0, 0, 0, 2*sigma_j2*dt, 2*sigma_j2*dt])
    return F, Q