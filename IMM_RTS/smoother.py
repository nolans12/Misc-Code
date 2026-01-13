from __future__ import annotations
from filterpy import common
import numpy as np
from filterpy.kalman import KalmanFilter
from utils import gaussian_pdf

class IMMSmoother:
    """
    Low-complexity IMM RTS smoother
    (Lopez & Danès, IEEE TAES 2017)

    Compatible with imm_filterpy.IMM.
    Uses imm.cache as the sole data source.
    """

    def __init__(self, PI: np.ndarray):
        self.PI = np.asarray(PI)
        self._cache: list[dict] = []

    @property
    def cache(self):
        """Read-only access to smoother cache"""
        return self._cache

    # ============================================================
    # RTS STEP (MODE-MATCHED SMOOTHING)
    # ============================================================
    @staticmethod
    def _rts_step(
        filter: KalmanFilter,
        x_filt: np.ndarray,
        P_filt: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        x_next: np.ndarray,
        P_next: np.ndarray,
        F: np.ndarray,
    ):
        """
        Standard Rauch–Tung–Striebel backward recursion.

        All inputs are in EXTERNAL space.
        RTS is performed in INTERNAL space and converted back.
        """
        # External → internal
        x_filt_i, P_filt_i = filter.to_internal(x_filt, P_filt)
        x_pred_i, P_pred_i = filter.to_internal(x_pred, P_pred)
        x_next_i, P_next_i = filter.to_internal(x_next, P_next)

        # RTS equations
            # for cpp, is it possilbe to recompute F? or just save it with the predictg and mu_mix?
        C = P_filt_i @ F.T
        G = C @ np.linalg.inv(P_pred_i)

        x_s_i = x_filt_i + G @ (x_next_i - x_pred_i)
        P_s_i = P_filt_i + G @ (P_next_i - P_pred_i) @ G.T

        # Internal → external
        x_s_e, P_s_e = filter.to_external(x_s_i, P_s_i)

        return x_s_e, P_s_e

    # ============================================================
    # MODE INTERACTION ENTIRE STEP
    # ============================================================
        
    def mode_interaction( 
        self,
        x_premixed: np.ndarray,
        P_premixed: np.ndarray,
        x_fwd: np.ndarray,
        P_fwd: np.ndarray,
        x_rts: np.ndarray,
        P_rts: np.ndarray,
        PI: np.ndarray,
        comm: float
    ):
        M = len(x_rts) # num of modes
        back_info_vectors = []
        back_info_matrices = []
        for i in range(M): 
            
            # DO ALL DATA FUSION W.R.T. COMMON FRAME
            
            P_rts_inv_i = np.linalg.inv(P_rts[i][0:comm, 0:comm]) 
            P_mixed_inv_i = np.linalg.inv(P_premixed[i][0:comm, 0:comm])
            
            back_info_matrices.append(P_rts_inv_i - P_mixed_inv_i) # 7, one-step backward predicted information matrix
            back_info_vectors.append((P_rts_inv_i @ x_rts[i][0:comm]) - (P_mixed_inv_i @ x_premixed[i][0:comm])) # 8, one-step backward predicted information vector      
         
        invertible = True
        for i in range(M):
            if np.linalg.matrix_rank(back_info_matrices[i]) != back_info_matrices[i].shape[0]:
                # This is checking if any of them arent invertible, in paper it does say really only care about i, but rather do this then dont have to renormalize
                invertible = False
                
        mu_mix, likelihoods = self.get_mixing_weights(back_info_vectors, back_info_matrices, x_fwd, P_fwd, PI, invertible, comm)
        return back_info_vectors, back_info_matrices, mu_mix, likelihoods, invertible
        
         
    def get_mixing_weights(self, Info_vectors, Info_matrices, x_fwd, P_fwd, PI, invertible, comm: float):
        if not invertible:
            # This is safe to have 0 p-mode likelihood because its just account for in bayes wiht += later\, so += 0*
            return PI, np.zeros_like(PI)
        else:
            M = len(x_fwd)
            x_back_preds = []
            P_back_preds = []
            for i in range(M):
                # Invert back to get x and Ps
                P_back_preds.append(np.linalg.inv(Info_matrices[i]))
                x_back_preds.append(P_back_preds[i] @ Info_vectors[i])

            mu_mix = np.zeros((M, M))
            likelihoods = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    # Get likelihood
                    Delta_ji = x_back_preds[i] - x_fwd[j][0:comm]
                    D_ji = P_back_preds[i] + P_fwd[j][0:comm,0:comm]
                    likelihood = gaussian_pdf(Delta_ji, np.zeros(len(Delta_ji)), D_ji) # 10, Two-mode conditioned likelihood
                    likelihoods[j, i] = max(likelihood, 1e-50)

            # Renormalize
            d = np.zeros(M)
            for j in range(M):
                d[j] = sum(PI[j, i] * likelihoods[j, i] for i in range(M))
            for i in range(M):
                for j in range(M):
                    mu_mix[i, j] = (PI[j, i] * likelihoods[j, i]) / d[j] # 11, Smoothed mixing probabilty
            return mu_mix, likelihoods
        
    def mode_interaction_1(self, M, x_fwd, P_fwd, back_info_vectors, back_info_matrices, mu_mix, comm: float):
        # Comput the fwd info mat and vec
        fwd_info_matrices = []
        fwd_info_vectors = []
        for i in range(M):
            fwd_info_matrices.append(np.linalg.inv(P_fwd[i][0:comm, 0:comm]))
            fwd_info_vectors.append(fwd_info_matrices[i] @ x_fwd[i][0:comm])
        
        # 1. Combination
        n = back_info_vectors[0].shape[0]
        P_jis = np.zeros((M, M, n, n))
        x_hat_jis =  np.zeros((M, M, n))
        for i in range(M): # 1
            for j in range(M): # 2
                # Two-mode conditioned smoothed estimate
                P_jis[j, i, :, :] = np.linalg.inv(back_info_matrices[i] + fwd_info_matrices[j]) # 3, Two-mode conditioned smoothed covariance
                x_hat_jis[j, i, :] = P_jis[j, i, :, :] @ (back_info_vectors[i] + fwd_info_vectors[j]) # 4, Two-mode conditioned smoothed mean

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
        return x_hats, P_hats

    def mode_interaction_2(self, M, x_fwd, P_fwd, back_state_vectors, back_cov_matrices, mu_mix, comm: float):
        # 1. Backward IMM mixing
        x_preds_back = []
        P_preds_back = []
        for j in range(M): # 1
            x_pred_back = sum(mu_mix[i, j] * back_state_vectors[i] for i in range(M))  # 2, Mixing backward predicted mean
            P_pred_back = sum(mu_mix[i, j] * (back_cov_matrices[i] + np.outer(back_state_vectors[i] - x_pred_back, back_state_vectors[i] - x_pred_back))  # 3, Mixing backward predicted covariance
                for i in range(M)
            )
            x_preds_back.append(x_pred_back)
            P_preds_back.append(P_pred_back)

        # 2. Combination
        x_hats_back = []
        P_hats_back = []
        for j in range(M): # 5
            P_pred_back_inv_j = np.linalg.inv(P_preds_back[j])
            P_hat_inv_j = np.linalg.inv(P_fwd[j][0:comm, 0:comm])
            P_hats_back.append(np.linalg.inv(P_pred_back_inv_j + P_hat_inv_j)) # 7, Mode-conditioned smoothed covariance
            x_hats_back.append(P_hats_back[j] @ ((P_pred_back_inv_j @ x_preds_back[j]) + (P_hat_inv_j @ x_fwd[j][0:comm]))) # 6, Mode-conditioned smoothed mean

        return x_hats_back, P_hats_back

    # ============================================================
    # MODE PROBABILITY SMOOTHING
    # ============================================================
    @staticmethod
    def _smooth_mode_probabilities(mu_fwd, mu_next, PI):
        """
        Equation (33) — smoothed mode probabilities
        """
        mu_pred = PI.T @ mu_fwd
        mu_bck = mu_pred * mu_next

        s = mu_bck.sum()
        if s > 0:
            mu_bck /= s
        else:
            mu_bck = mu_fwd.copy()

        return mu_bck
        
    def get_smoothed_pmodes(self, mu, PI, likelihoods, invertible):
        # instead of passing in invertible, just check if d == 0?
        # if it isnt invertible the likelihoods are 0 and it adds no information to bayes anyways
        M = mu.shape[0]
        if invertible:
            d = np.zeros(M)
            for j in range(M):
                for i in range(M):
                    d[j] += PI[j, i] * likelihoods[j, i]
            mu_bck = np.zeros(M)
            den = mu.T @ d
            for j in range(M):
                mu_bck[j] = d[j] * mu[j] / den
            return mu_bck
        else:
            return mu

    def fuse_states_and_covs(self, M, mus, xs, Ps):
        x_hat = sum(mus[j] * xs[j] for j in range(M))
        n = Ps[0].shape[0]
        P_hat = np.zeros((n,n))
        for j in range(M):
            P_hat += mus[j] * (Ps[j] + np.outer(xs[j] - x_hat, xs[j] - x_hat))
        return x_hat, P_hat

    # ============================================================
    # MAIN SMOOTHER
    # ============================================================
    def smooth(self, imm) -> None:
        """
        Run fixed-interval IMM smoothing.

        Results are stored in self.cache (like IMM).
        """

        self._cache.clear()

        cache = imm.cache
        T = len(cache)
        M = imm.M
        n = imm.n

        # Allocate backward arrays
        x_mode = np.zeros((T, M, n))
        P_mode = np.zeros((T, M, n, n))
        mu_s   = np.zeros((T, M))

        # --------------------------------------------------------
        # INITIALIZATION (k = T-1)
        # --------------------------------------------------------
        last = cache[-1]
        for j in range(M):
            x_mode[-1, j] = last["x_filter"][j]
            P_mode[-1, j] = last["P_filter"][j]
        mu_s[-1] = last["mu"]
        
        # Copy first snapshot in 
        snapshot = {
            "x_s": last['x_fuse'][0:imm.common_dim],
            "P_s": last['P_fuse'][0:imm.common_dim,0:imm.common_dim],
            "x_filt_s": last['x_filter'],
            "P_filt_s": last['P_filter'],
            "mu_s": last["mu"],
        }
        self.cache.append(snapshot)


        # --------------------------------------------------------
        # BACKWARD PASS
        # --------------------------------------------------------
        for k in reversed(range(T)):
            if k < T - 1:
                cur = cache[k]
                nxt = cache[k + 1]
                PI = nxt["PI"]  # dt-scaled TPM from filtering, from cur to nxt.

                # ------------------------
                # MODE-MATCHED RTS
                # ------------------------
                x_rts = np.zeros((M, n))
                P_rts = np.zeros((M, n, n))

                for j in range(M):
                    x_rts[j], P_rts[j] = self._rts_step(
                        imm.filters[j],
                        cur["x_filter"][j],
                        cur["P_filter"][j],
                        nxt["x_pred"][j],
                        nxt["P_pred"][j],
                        x_mode[k + 1, j],
                        P_mode[k + 1, j],
                        cur["F"][j],
                    )

                a = 1
                # ------------------------
                # MODE INTERACTION
                # ------------------------
                back_info_vectors, back_info_matrices, mu_mix, likelihoods, invertible = self.mode_interaction(
                    nxt["x_premixed"],
                    nxt["P_premixed"],
                    x_rts,
                    P_rts,
                    cur['x_filter'],
                    cur['P_filter'],
                    PI,
                    imm.common_dim
                )

                if invertible:
                    # METHOD 2
                    back_state_vectors = []
                    back_cov_matrices = []
                    for j in range(M):
                        back_cov_matrices.append(np.linalg.inv(back_info_matrices[j]))
                        back_state_vectors.append(back_cov_matrices[j] @ back_info_vectors[j])
                    x_filt_smooth, P_filt_smooth = self.mode_interaction_2(M, cur['x_filter'], cur['P_filter'], back_state_vectors, back_cov_matrices, mu_mix, imm.common_dim)
                else:
                    # METHOD 1 
                    x_filt_smooth, P_filt_smooth = self.mode_interaction_1(M, cur['x_filter'], cur['P_filter'], back_info_vectors, back_info_matrices, mu_mix, imm.common_dim)
        
                # ------------------------
                # MODE PROBABILITY SMOOTHING
                # ------------------------
                
                need to figure out what to do with common dim state, otherwise x_smooth is 6d,
                probably take from common and then add full accel at end
                
                like if size > common
                take common : end and append it 
                
                # ensure use PI from next cache, not curr
                mu_smooth = self.get_smoothed_pmodes(cur["mu"], PI, likelihoods, invertible)
                x_smooth, P_smooth = self.fuse_states_and_covs(M, mu_smooth, x_filt_smooth, P_filt_smooth)
            
                # Save x_mode, P_mode
                x_mode[k] = x_filt_smooth
                P_mode[k] = P_filt_smooth

                # ----------------------------------------------------
                # SNAPSHOT 
                # ----------------------------------------------------
                snapshot = {
                    "x_s": x_smooth,
                    "P_s": P_smooth,
                    "x_filt_s": x_filt_smooth,
                    "P_filt_s": P_filt_smooth,
                    "mu_s": mu_smooth,
                }

                self.cache.append(snapshot)


        # At the end reverse snapshots
        self._cache = self._cache[::-1]