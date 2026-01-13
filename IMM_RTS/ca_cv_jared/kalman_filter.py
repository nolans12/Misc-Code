import numpy as np

def kf_predict(x, P, F, Q):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def kf_update(x_pred, P_pred, z, H, R):
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    y = z - H @ x_pred
    x_f = x_pred + K @ y
    P_f = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_f, P_f