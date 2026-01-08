# smoother.py
from __future__ import annotations
import numpy as np


def sym(P):
    return 0.5 * (P + P.T)


class IMMSmootherRTS:
    """
    Interacting Multiple Model RTS smoother.

    Assumes:
      - Forward IMM has been run
      - imm._history exists
      - Each model provides:
          to_common(x, P)
          to_internal(x_common, P_common)
    """

    def __init__(self, PI: np.ndarray, eps: float = 1e-12):
        self.PI = np.asarray(PI, dtype=float)
        self.eps = eps

    def smooth(self, imm):
        hist = imm._history
        K = len(hist)
        M = len(hist[0]["models"])
        common_dim = imm.common_dim

        # --- Storage ---
        x_mode = [[None]*M for _ in range(K)]
        P_mode = [[None]*M for _ in range(K)]
        mu_s = np.zeros((K, M))

        x_common = np.zeros((K, common_dim, 1))
        P_common = np.zeros((K, common_dim, common_dim))

        # ==========================================================
        # Initialize at final time (smoothed = filtered)
        # ==========================================================
        mu_s[-1] = hist[-1]["mu"]

        for j, model in enumerate(imm.models):
            x_f, P_f = hist[-1]["models"][j]["x_filt"], hist[-1]["models"][j]["P_filt"]
            x_mode[-1][j] = x_f.copy()
            P_mode[-1][j] = P_f.copy()

        x_common[-1], P_common[-1] = self._moment_match(
            imm, x_mode[-1], P_mode[-1], mu_s[-1]
        )

        # ==========================================================
        # Backward recursion
        # ==========================================================
        for k in range(K-2, -1, -1):

            # print(k)
            # print(f"time: {hist[k]['time']}")

            mu_f = hist[k]["mu"]
            mu_next = mu_s[k+1]

            # ---------- backward transition probs b_ij ----------
            c = mu_f @ self.PI # this should be forward pass
            c = np.maximum(c, self.eps)

            # print(f"curr_p_mode: {c}")
            # print(f"pred_p_trans: {self.PI}")

            b = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    b[i, j] = self.PI[j, i] * mu_f[j] / c[i]

            # print(f"backward_trans_prob: {b}")

            # ---------- backward mixing ----------
            d = b.T @ mu_next
            d = np.maximum(d, self.eps)

            mu_mix = np.zeros((M, M))
            for j in range(M):
                for i in range(M):
                    mu_mix[i, j] = b[i, j] * mu_next[i] / d[j]

            # print(f"backward_mixing: {mu_mix}")

            # ---------- mix smoothed states (COMMON space) ----------
            x0 = [np.zeros((common_dim, 1)) for _ in range(M)]
            P0 = [np.zeros((common_dim, common_dim)) for _ in range(M)]

            for j in range(M):
                for i in range(M):
                    # print(f"xi_c: {i}: {x_mode[k + 1][i]}")
                    xi_c, Pi_c = imm.models[i].to_common(
                        x_mode[k+1][i], P_mode[k+1][i]
                    )
                    x0[j] += mu_mix[i, j] * xi_c

                for i in range(M):
                    xi_c, Pi_c = imm.models[i].to_common(
                        x_mode[k+1][i], P_mode[k+1][i]
                    )
                    dx = xi_c - x0[j]
                    P0[j] += mu_mix[i, j] * (Pi_c + dx @ dx.T)
                # print(f"mixed_estimate.state: {x0[j]}")
                # print(f"mixed_estimate.covariance: {P0[j]}")
                # print(f"mixed_estimate.covariance determinate: {np.linalg.det(P0[j])}")

            # ---------- RTS per model (INTERNAL space) ----------
            for j, model in enumerate(imm.models):
                snap = hist[k]["models"][j]
                snap_p = hist[k+1]["models"][j]

                x_f = snap["x_filt"]
                P_f = snap["P_filt"]
                x_pred = snap_p["x_pred"]
                P_pred = snap_p["P_pred"]
                F = snap_p["F"]

                # Convert mixed common -> internal
                x0_j, P0_j = model.to_internal(x0[j], P0[j])

                # RTS gain
                P_pred_inv = np.linalg.pinv(P_pred)
                A = P_f @ F.T @ P_pred_inv

                x_s = x_f + A @ (x0_j - x_pred)
                P_s = P_f - A @ (P0_j - P_pred) @ A.T

                x_mode[k][j] = x_s
                P_mode[k][j] = sym(P_s)

                # print(f"smooth_snapshot.filter_data.state: {x_s}")
            # ---------- smoothed mode probabilities ----------

            # USE INTERNAL
            Lambda = np.zeros(M)
            for j in range(M):
                val = 0.0
                x_pred_c, P_pred_c = imm.models[j].to_common(
                    hist[k+1]["models"][j]["x_pred"],
                    hist[k+1]["models"][j]["P_pred"],
                )
                for i in range(M):
                    xi_c, _ = imm.models[i].to_common(
                        x_mode[k+1][i], P_mode[k+1][i]
                    )
                    y = xi_c - x_pred_c
                    # print(f"y: {y}")
                    val += self.PI[j, i] * self._gauss(y[0:6], P_pred_c[0:6,0:6])
                    # print(f"value iter: {self.PI[j, i] * self._gauss(y[0:6], P_pred_c[0:6,0:6])}")
                Lambda[j] = max(val, self.eps)

            # print(f"smoothed_likelihoods: {Lambda}")
            mu_tmp = Lambda * mu_f
            mu_s[k] = mu_tmp / np.sum(mu_tmp)
            print(f"at time: {hist[k]['time']:.1f}\n"
                  f"forward p modes: {c}\n"
                  f"smoothed p modes: {mu_s[k]}\n")

            # ---------- moment match ----------
            x_common[k], P_common[k] = self._moment_match(
                imm, x_mode[k], P_mode[k], mu_s[k]
            )

        return {
            "x_s": x_common,
            "P_s": P_common,
            "mu": mu_s,
            "x_mode": x_mode,
            "P_mode": P_mode,
        }

    # ==============================================================
    # Helpers
    # ==============================================================

    def _moment_match(self, imm, x_modes, P_modes, mu):
        x = np.zeros((imm.common_dim, 1))
        P = np.zeros((imm.common_dim, imm.common_dim))

        for j, model in enumerate(imm.models):
            xj, Pj = model.to_common(x_modes[j], P_modes[j])
            x += mu[j] * xj

        for j, model in enumerate(imm.models):
            xj, Pj = model.to_common(x_modes[j], P_modes[j])
            dx = xj - x
            P += mu[j] * (Pj + dx @ dx.T)

        return x, sym(P)

    @staticmethod
    def _gauss(y, S):
        try:
            L = np.linalg.cholesky(S)
            v = np.linalg.solve(L, y)
            maha = v.T @ v
            log_det = 2 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            maha = y.T @ S_inv @ y
            sign, log_det = np.linalg.slogdet(S)
        n = y.shape[0]
        return float(np.exp(-0.5 * (n*np.log(2*np.pi) + log_det + maha)))
