import numpy as np

def cv_model(dt, sigma_a):
    F = np.array([
        [1, dt, 0.5*dt**2],
        [0, 1,  dt],
        [0, 0,  0]
    ])
    Q = np.diag([0, 0, sigma_a**2])
    return F, Q

def cv_model_2d(dt, sigma_a2):
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Q = np.diag([0, 0, 2*sigma_a2*dt, 2*sigma_a2*dt])
    return F, Q

def ca_model(dt, sigma_j):
    F = np.array([
        [1, dt, 0.5*dt**2],
        [0, 1,  dt],
        [0, 0,  1]
    ])
    Q = np.diag([0, 0, sigma_j**2])
    return F, Q