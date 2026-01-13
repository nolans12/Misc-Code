import numpy as np

from kalman_filter import kf_predict, kf_update
from utils import gaussian_pdf


def imm_filter(zs, models, Pi, x0s, P0s, mu0):
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

        for j in range(M):
            xbar = sum(mu_ij[i, j] * x_f[i] for i in range(M))
            Pbar = sum(
                mu_ij[i, j] * (
                    P_f[i]
                    + np.outer(x_f[i] - xbar, x_f[i] - xbar)
                )
                for i in range(M)
            )
            x_mix.append(xbar)
            P_mix.append(Pbar)

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
            "x_premixed": x_mix,
            "P_premixed": P_mix,
            "x_hats": x_new,
            "P_hats": P_new,
            "x_preds": x_pred,
            "P_preds": P_pred,
            "C": C,
            "mu": mu
        })

        x_f, P_f = x_new, P_new

    return forward