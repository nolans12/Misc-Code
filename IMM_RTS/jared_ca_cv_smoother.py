# Hacked version of jared ca_cv smoother with dimension preserving etc.
# Also changes input / output so can just plug and play with my existing attempts

import numpy as np
from utils import gaussian_pdf, log_gaussian_pdf


class IMMSmoother:
    def __init__(self, PI: np.ndarray):
        self.PI = np.asarray(PI)
        self._cache = []

    @property
    def cache(self):
        return self._cache

    # ============================================================
    # MODE-MATCHED RTS (UNCHANGED MATH)
    # ============================================================
    @staticmethod
    def mode_matched_smoothing(next_fwd, bck):
        x_bars    = next_fwd["x_bars"]
        P_bars    = next_fwd["P_bars"]
        x_preds   = next_fwd["x_preds"]
        P_preds   = next_fwd["P_preds"]
        Cross_cov = next_fwd["C"]

        x_smooth_prev = bck["x_hats"]
        P_smooth_prev = bck["P_hats"]

        x_rts, P_rts = [], []
        for i in range(len(x_bars)):
            G = Cross_cov[i] @ np.linalg.inv(P_preds[i])
            xb = x_bars[i][0:len(x_preds[i])] + G @ (x_smooth_prev[i] - x_preds[i])
            Pb = P_bars[i][0:len(x_preds[i]), 0:len(x_preds[i])] + G @ (P_smooth_prev[i] - P_preds[i]) @ G.T
            x_rts.append(xb)
            P_rts.append(Pb)
        return x_rts, P_rts

    # ============================================================
    # MIXING WEIGHTS (UNCHANGED MATH)
    # ============================================================
    @staticmethod
    def get_mixing_weights(Pi, Info_matrices, Info_vectors, x_hats, P_hats, invertible):
        if not invertible:
            return Pi, np.zeros_like(Pi)

        M = len(x_hats)
        x_back_preds, P_back_preds = [], []
        for i in range(M):
            P_back_preds.append(np.linalg.inv(Info_matrices[i]))
            x_back_preds.append(P_back_preds[i] @ Info_vectors[i])

        mu_mix = np.zeros((M, M))
        likelihoods = np.zeros((M, M))

        # Shared state extraction (UNCHANGED)
        x_back_preds_shared = x_back_preds.copy()
        x_back_preds_shared[0] = x_back_preds[0][0:6]
        P_back_preds_shared = P_back_preds.copy()
        P_back_preds_shared[0] = P_back_preds[0][0:6, 0:6]

        x_hats_shared = x_hats.copy()
        x_hats_shared[0] = x_hats[0][0:6]
        P_hats_shared = P_hats.copy()
        P_hats_shared[0] = P_hats[0][0:6, 0:6]

        for i in range(M):
            for j in range(M):
                Delta_ji = x_back_preds_shared[i] - x_hats_shared[j]
                D_ji = P_back_preds_shared[i] + P_hats_shared[j]
                likelihood = gaussian_pdf(
                    Delta_ji, np.zeros(len(Delta_ji)), D_ji
                )
                likelihoods[j, i] = max(likelihood, 1e-12)
                print(likelihoods)

        d = np.zeros(M)
        for j in range(M):
            d[j] = sum(Pi[j, i] * likelihoods[j, i] for i in range(M))

        for i in range(M):
            for j in range(M):
                mu_mix[i, j] = (Pi[j, i] * likelihoods[j, i]) / d[j]
        
        return mu_mix, likelihoods
        
        # log_likelihoods = np.zeros((M, M))

        # # --------------------------------------------------
        # # Compute log-likelihoods
        # # --------------------------------------------------
        # for i in range(M):
        #     for j in range(M):
        #         Delta_ji = x_back_preds_shared[i] - x_hats_shared[j]
        #         D_ji = P_back_preds_shared[i] + P_hats_shared[j]

        #         log_likelihoods[j, i] = log_gaussian_pdf(
        #             Delta_ji,
        #             np.zeros(len(Delta_ji)),
        #             D_ji
        #         )

        # # --------------------------------------------------
        # # Convert to likelihoods (SAFE)
        # # --------------------------------------------------
        # likelihoods = np.zeros_like(log_likelihoods)
        
        # row_max = np.max(log_likelihoods, axis=1, keepdims=True)
        # bad_rows = ~np.isfinite(row_max).squeeze()

        # # Normal rows
        # good = ~bad_rows
        # likelihoods[good] = np.exp(
        #     log_likelihoods[good] - row_max[good]
        # )

        # # Bad rows → uniform likelihood (no information)
        # likelihoods[bad_rows] = 1.0

        # # optional floor (purely defensive)
        # likelihoods = np.maximum(likelihoods, 1e-300)

        # # --------------------------------------------------
        # # Compute normalization
        # # --------------------------------------------------
        # d = np.zeros(M)
        # for j in range(M):
        #     d[j] = np.sum(Pi[j, :] * likelihoods[j, :])

        # # --------------------------------------------------
        # # Compute mixing probabilities
        # # --------------------------------------------------
        # mu_mix = np.zeros((M, M))
        # for i in range(M):
        #     for j in range(M):
        #         mu_mix[i, j] = (Pi[j, i] * likelihoods[j, i]) / d[j]


        # return mu_mix, likelihoods

    # ============================================================
    # MODE INTERACTION PRELIMINARY (UNCHANGED MATH)
    # ============================================================
    def mode_interaction_preliminary(self, M, Pi, fwd, next_fwd, x_rts, P_rts):
        x_hats = fwd["x_hats"]
        P_hats = fwd["P_hats"]
        x_bars = next_fwd["x_bars"]
        P_bars = next_fwd["P_bars"]

        back_info_matrices, back_info_vectors = [], []
        fwd_info_matrices, fwd_info_vectors = [], []

        for i in range(M):
            P_rts_inv = np.linalg.inv(P_rts[i])
            P_bar_inv = np.linalg.inv(P_bars[i][0:len(x_rts[i]), 0:len(x_rts[i])])

            fwd_info_matrices.append(np.linalg.inv(P_hats[i]))
            fwd_info_vectors.append(fwd_info_matrices[i] @ x_hats[i])

            back_info_matrices.append(P_rts_inv - P_bar_inv[0:len(x_rts[i]), 0:len(x_rts[i])])
            back_info_vectors.append((P_rts_inv @ x_rts[i]) - (P_bar_inv @ x_bars[i][0:len(x_rts[i])]))

        try:
            invertible = all(
                np.linalg.matrix_rank(B) == B.shape[0] for B in back_info_matrices
            )
        except np.linalg.LinAlgError as e:
            if "SVD did not converge" in str(e):
                invertible = False
            else:
                raise

        mu_mix, likelihoods = self.get_mixing_weights(
            Pi, back_info_matrices, back_info_vectors, x_hats, P_hats, invertible
        )

        return (
            fwd_info_vectors,
            fwd_info_matrices,
            back_info_vectors,
            back_info_matrices,
            mu_mix,
            likelihoods,
            invertible,
        )

    # ============================================================
    # MODE INTERACTION 1
    # ============================================================
    def mode_interaction_1(self, M, fwd_info_vectors, fwd_info_matrices, back_info_vectors, back_info_matrices, mu_mix):
        # 1. Combination
        n = back_info_vectors[0].shape[0]
        P_jis = np.zeros((M, M, n, n))
        x_hat_jis =  np.zeros((M, M, n))

        # Extract Shared Statespace
        fwd_info_vectors_shared = fwd_info_vectors.copy()
        fwd_info_vectors_shared[0] = fwd_info_vectors[0][0:6]
        fwd_info_matrices_shared = fwd_info_matrices.copy()
        fwd_info_matrices_shared[0] = fwd_info_matrices[0][0:6, 0:6]
        back_info_vectors_shared = back_info_vectors.copy()
        back_info_vectors_shared[0] = back_info_vectors[0][0:6]
        back_info_matrices_shared = back_info_matrices.copy()
        back_info_matrices_shared[0] = back_info_matrices[0][0:6, 0:6]

        # Two-mode conditioned smoothed estimates
        # CA - CA: Full fusion (9D)
        P_jis[0, 0, :, :] = np.linalg.inv(back_info_matrices[0] + fwd_info_matrices[0]) # 3, Two-mode conditioned smoothed covariance
        x_hat_jis[0, 0, :] = P_jis[0, 0, :, :] @ (back_info_vectors[0] + fwd_info_vectors[0]) # 4, Two-mode conditioned smoothed mean

        # CA - CV: Shared space and then re-insert accels
        P_jis_01 = np.linalg.inv(back_info_matrices_shared[1] + fwd_info_matrices_shared[0]) # 3, Two-mode conditioned smoothed covariance
        x_hat_jis_01 = P_jis_01 @ (back_info_vectors_shared[1] + fwd_info_vectors_shared[0]) # 4, Two-mode conditioned smoothed mean
        x_hat_jis[0, 1, :] = np.concatenate([x_hat_jis_01, x_hat_jis[0, 0, 6:9]])
        P_jis[0, 1, :, :] = np.block([
            [P_jis_01, P_jis[0, 0, 0:6, 6:9]],
            [P_jis[0, 0, 6:9, :]]])

        # CV - CA: Shared space and then re-insert accels
        P_jis_10 = np.linalg.inv(back_info_matrices_shared[0] + fwd_info_matrices_shared[1]) # 3, Two-mode conditioned smoothed covariance
        x_hat_jis_10 = P_jis_10 @ (back_info_vectors_shared[0] + fwd_info_vectors_shared[1]) # 4, Two-mode conditioned smoothed mean
        x_hat_jis[1, 0, :] = np.concatenate([x_hat_jis_10, x_hat_jis[0, 0, 6:9]])
        P_jis[1, 0, :, :] = np.block([
            [P_jis_10, P_jis[0, 0, 0:6, 6:9]],
            [P_jis[0, 0, 6:9, :]]])

        # CV - CV: Shared space and then re-insert accels
        P_jis_11 = np.linalg.inv(back_info_matrices_shared[1] + fwd_info_matrices_shared[1]) # 3, Two-mode conditioned smoothed covariance
        x_hat_jis_11 = P_jis_11 @ (back_info_vectors_shared[1] + fwd_info_vectors_shared[1]) # 4, Two-mode conditioned smoothed mean
        x_hat_jis[1, 1, :] = np.concatenate([x_hat_jis_11, x_hat_jis[0, 0, 6:9]])
        P_jis[1, 1, :, :] = np.block([
            [P_jis_11, P_jis[0, 0, 0:6, 6:9]],
            [P_jis[0, 0, 6:9, :]]])

        # 2. Mixing
        x_hats = []
        P_hats = []
        for j in range(M): # 7
            x_hat = sum(mu_mix[i, j] * x_hat_jis[j, i, :] for i in range(M)) # 8, Mode-conditioned smoothed mean
            P_hat = sum(
                mu_mix[i, j] * (
                        P_jis[j, i, :, :]
                        + np.outer(x_hat_jis[j, i, :] - x_hat, x_hat_jis[j, i, :] - x_hat)
                ) # 9, Mode-conditioned smoothed covariance
                for i in range(M)
            )
            x_hats.append(x_hat)
            P_hats.append(P_hat)
        # Remove accel components from CV model state / cov
        x_hats[1] = x_hats[1][0:6]
        P_hats[1] = P_hats[1][0:6, 0:6]
        return x_hats, P_hats

    # ============================================================
    # MODE PROBABILITY SMOOTHING (UNCHANGED MATH)
    # ============================================================
    @staticmethod
    def get_smoothed_pmodes(fwd, Pi, likelihoods, invertible):
        mu_fwd = fwd["mu"]
        M = mu_fwd.shape[0]

        if not invertible:
            return mu_fwd

        d = np.zeros(M)
        for j in range(M):
            for i in range(M):
                d[j] += Pi[j, i] * likelihoods[j, i]

        mu_bck = np.zeros(M)
        den = mu_fwd.T @ d
        for j in range(M):
            mu_bck[j] = d[j] * mu_fwd[j] / den
        return mu_bck

    # ============================================================
    # FUSION 
    # ============================================================
    def fuse_states_and_covs(self, M, mus, xs, Ps):
        # Extract Shared Statespace
        xs_shared = xs.copy()
        xs_shared[0] = xs[0][0:6]
        Ps_shared = Ps.copy()
        Ps_shared[0] = Ps[0][0:6, 0:6]

        x_hat = sum(mus[j] * xs_shared[j] for j in range(M))
        n = Ps_shared[0].shape[0]
        P_hat = np.zeros((n,n))

        for j in range(M):
            P_hat += mus[j] * (Ps_shared[j] + np.outer(xs_shared[j] - x_hat, xs_shared[j] - x_hat))

        x_hat = np.concatenate([x_hat, xs[0][6:9]])
        P_hat = np.block([
            [P_hat, Ps[0][0:6, 6:9]],
            [Ps[0][6:9, :]]])
        return x_hat, P_hat


    # ============================================================
    # MAIN ENTRY POINT (MATCHES YOUR CALL)
    # ============================================================
    def smooth(self, imm):
        self._cache.clear()
        cache = imm.cache
        T = len(cache)
        M = imm.M
        PI = self.PI

        # init
        backward = [{
            "x_hats": cache[-1]["x_filter"],
            "P_hats": cache[-1]["P_filter"],
            "mu": cache[-1]["mu"],
        }]

        self._cache.append({
            "x_s": cache[-1]["x_fuse"],
            "P_s": cache[-1]["P_fuse"],
            "x_filt_s": cache[-1]["x_filter"],
            "P_filt_s": cache[-1]["P_filter"],
            "mu_s": cache[-1]["mu"],
        })

        for k in reversed(range(T - 1)):
            bck = backward[-1]
            fwd = {
                "x_hats": cache[k]["x_filter"],
                "P_hats": cache[k]["P_filter"],
                "mu": cache[k]["mu"],
            }
            next_fwd = {
                "x_bars": cache[k + 1]["x_premixed"],
                "P_bars": cache[k + 1]["P_premixed"],
                "x_preds": cache[k + 1]["x_pred"],
                "P_preds": cache[k + 1]["P_pred"],
                "C": cache[k+1]["C"],
            }

            x_rts, P_rts = self.mode_matched_smoothing(next_fwd, bck)

            (
                fwd_info_vectors,
                fwd_info_matrices,
                back_info_vectors,
                back_info_matrices,
                mu_mix,
                likelihoods,
                invertible,
            ) = self.mode_interaction_preliminary(
                M, PI, fwd, next_fwd, x_rts, P_rts
            )

            x_hats_bck, P_hats_bck = self.mode_interaction_1(
                M, fwd_info_vectors, fwd_info_matrices,
                back_info_vectors, back_info_matrices, mu_mix
            )
            # no mode interaction 2 rn

            mu_bck = self.get_smoothed_pmodes(fwd, PI, likelihoods, invertible)
            x_s, P_s = self.fuse_states_and_covs(M, mu_bck, x_hats_bck, P_hats_bck)

            backward.append({
                "x_hats": x_hats_bck,
                "P_hats": P_hats_bck,
                "mu": mu_bck,
            })

            self._cache.append({
                "x_s": x_s,
                "P_s": P_s,
                "x_filt_s": x_hats_bck,
                "P_filt_s": P_hats_bck,
                "mu_s": mu_bck,
            })

        self._cache.reverse()
