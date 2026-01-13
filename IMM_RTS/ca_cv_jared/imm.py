import numpy as np

from kalman_filter import kf_predict, kf_update
from utils import gaussian_pdf


def imm_filter_ca_cv(zs, models, Pi, x0s, P0s, mu0):
    """
    models[i] = (F_i, Q_i, H_i, R_i)
    """
    M = len(models)
    T = len(zs)

    forward = []

    x_f = x0s
    P_f = P0s
    mu = mu0

    for k in range(T):
        # --- Mixing ---
        c_j = Pi.T @ mu
        mu_ij = (Pi * mu[:, None]) / c_j[None, :]

        x_mix = []
        P_mix = []

        # Extract Shared Statespace
        x_f_shared = x_f.copy()
        x_f_shared[0] = x_f[0][0:4]
        P_f_shared = P_f.copy()
        P_f_shared[0] = P_f[0][0:4, 0:4]
        # Mixed Shared Statespace Only
        for j in range(M):
            xbar = sum(mu_ij[i, j] * x_f_shared[i] for i in range(M))
            Pbar = sum(
                mu_ij[i, j] * (
                    P_f_shared[i] + np.outer(x_f_shared[i] - xbar, x_f_shared[i] - xbar)
                )
                for i in range(M)
            )
            x_mix.append(xbar)
            P_mix.append(Pbar)
        # Re-insert Augmented Components
        x_mix[0] = np.concatenate([x_mix[0], x_f[0][4:6]])
        P_mix[0] = np.block([
            [P_mix[0], P_f[0][0:4, 4:6]],
            [P_f[0][4:6,:]] ])

        # --- Prediction + Update ---
        x_pred, P_pred = [], []
        x_new, P_new = [], []
        likelihood = np.zeros(M)

        for i, (F, Q, H, R) in enumerate(models):
            xp, Pp = kf_predict(x_mix[i], P_mix[i], F, Q)
            xf, Pf = kf_update(xp, Pp, zs[k], H, R)

            x_pred.append(xp)
            P_pred.append(Pp)
            x_new.append(xf)
            P_new.append(Pf)

            likelihood[i] = gaussian_pdf(zs[k], H @ xp, H @ Pp @ H.T + R)

        mu = c_j * likelihood
        mu /= np.sum(mu)

        # RTS cross-covariance
        C = [P_mix[i] @ models[i][0].T for i in range(M)]

        forward.append({
            "x_hats": x_new,
            "P_hats": P_new,
            "x_bars": x_mix,
            "P_bars": P_mix,
            "x_preds": x_pred,
            "P_preds": P_pred,
            "C": C,
            "mu": mu
        })

        x_f, P_f = x_new, P_new

    return forward