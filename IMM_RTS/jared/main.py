import numpy as np

from imm import imm_filter
from models import cv_model, ca_model, cv_model_2d
from lopez_danes_smoother import smooth

import plotly.io as pio
pio.renderers.default = "browser"

import pandas as pd
import plotly.express as px

def make_models(test_case):
    # --- Create Models ---
    dt = 0.1

    # 1D CA - CA example
    models = []
    x0s = []
    P0s = []
    if test_case == 1:
        F_CA_1, Q_CA_1 = ca_model(dt, 0.05)
        F_CA_2, Q_CA_2 = ca_model(dt, 10.0)
        H = np.array([[1.0, 0.0, 0.0]])
        R = np.array([[0.2]])
        models = [
            (F_CA_1, Q_CA_1, H, R),   # Model 0: smooth motion
            (F_CA_2, Q_CA_2, H, R)    # Model 1: maneuvering
        ]
        x0s = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        ]

        P0s = [
            np.diag([1.0, 1.0, 1.0]),
            np.diag([1.0, 1.0, 1.0]),
        ]
    # 2D CV - CV example
    elif test_case == 2:
        F_CV_2d_1, Q_CV_2d_1 = cv_model_2d(dt, 0.005)
        F_CV_2d_2, Q_CV_2d_2 = cv_model_2d(dt, 50.0)
        H_2d = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]])
        R_2d = np.diag([0.2, 0.2])
        models = [
            (F_CV_2d_1, Q_CV_2d_1, H_2d, R_2d),   # Model 0: smooth motion
            (F_CV_2d_2, Q_CV_2d_2, H_2d, R_2d)    # Model 1: maneuvering
        ]
        x0s = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0])
        ]

        P0s = [
            np.diag([100.0, 100.0, 10.0, 10.0]),
            np.diag([100.0, 100.0, 10.0, 10.0]),
        ]
    else:
        ValueError('Invalid test case')

    mu0 = np.array([0.5, 0.5])
    Pi = np.array([
        [0.99, 0.01],
        [0.01, 0.99]
    ])
    return models, x0s, P0s, mu0, Pi

def simulate(T, models, n_steps):
    F0, Q0, H0, R0 = models[0]
    m = H0.shape[0]
    n = F0.shape[0]
    x = np.zeros(n)
    xs = []
    zs = []
    modes = []

    for k in range(T):
        # Mode switch halfway
        mode = 0
        if n_steps == 3:
            if T // 4 < k < T // 2 or k > 3 * T // 4:
                mode = 1
        elif n_steps == 2:
            if T // 3 < k < 2 * T // 2:
                mode = 1
        elif n_steps == 1:
            if k > T // 2:
                mode = 1
        else:
            ValueError('Set a valid n_steps parameter')

        F, Q, H, R = models[mode]

        w = np.random.multivariate_normal(np.zeros(n), Q)
        v = np.random.multivariate_normal(np.zeros(m), R)

        x = F @ x + w
        z = H @ x + v

        xs.append(x)
        zs.append(z)
        modes.append(mode)

    return np.array(xs), np.array(zs), np.array(modes)


def main():
    # --- Config Inputs --- #
    n_steps = 3
    T = 120
    test_case = 2
    interaction_mode = 2
    np.random.seed(2)

    # --- Simulate Target - --
    models, x0s, P0s, mu0, Pi = make_models(test_case)
    xs_true, zs, modes_true = simulate(T, models, n_steps)

    # --- Set p-modes ---
    p0_true = []
    p1_true = []
    for k in range(T):
        if modes_true[k] == 0:
            p0_true.append(1.0)
            p1_true.append(0.0)
        else:
            p0_true.append(0.0)
            p1_true.append(1.0)

    forward = imm_filter(zs, models, Pi, x0s, P0s, mu0)
    backward = smooth(forward, Pi, interaction_mode)

    # --- Diagnostics ---
    mu_filts = np.array([f["mu"] for f in forward])
    mu_smooth = np.array([b["mu"] for b in backward])

    # RMSE comparison
    x_filt_mean = np.array([
        sum(f["mu"][i] * f["x_hats"][i] for i in range(2))
        for f in forward
    ])

    x_smooth_mean = np.array([
        sum(b["mu"][i] * b["x_hats"][i] for i in range(2))
        for b in backward
    ])

    rmse_filt = np.sqrt(np.mean((x_filt_mean[:,0] - xs_true[:,0])**2))
    rmse_smooth = np.sqrt(np.mean((x_smooth_mean[:,0] - xs_true[:,0])**2))

    print("RMSE filtered:", rmse_filt)
    print("RMSE smoothed:", rmse_smooth)

    df_state = pd.DataFrame({
        "time": np.arange(T),
        "p0_true": p0_true,
        "p0_filt": mu_filts[:, 0],
        "p0_smooth": mu_smooth[:, 0],
    })

    fig1 = px.line(
        df_state,
        x="time",
        y=["p0_true", "p0_filt", "p0_smooth"],
        title="IMM State Estimate (Position)",
        labels={"value": "p mode", "variable": "p mode"}
    )
    fig1.show()

if __name__ == "__main__":
    main()
