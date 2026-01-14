# THIS JUST DOES EVERYTHING IN FULL 9D FRAME AND ASSUMES THE TO EXTERNAL TAKES CARE OF ISSUES


from __future__ import annotations
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
        x_filt: np.ndarray, # curr
        P_filt: np.ndarray,
        x_pred: np.ndarray, # prediction 
        P_pred: np.ndarray,
        x_next: np.ndarray, # goal
        P_next: np.ndarray,
        F: np.ndarray, # from next
    ):
        """
        Standard Rauch–Tung–Striebel backward recursion.

        All inputs should be internal for RTS
        RTS is performed in INTERNAL space and outputted to INTERNAL
        """
        # RTS equations
            # for cpp, is it possilbe to recompute F? or just save it with the predictg and mu_mix?
        C = P_filt @ F.T
        G = C @ np.linalg.inv(P_pred)

        x_s = x_filt + G @ (x_next - x_pred)
        P_s = P_filt + G @ (P_next - P_pred) @ G.T

        # print(f"P_s: \n{P_s[3:6,3:6]}")
        return x_s, P_s

    # ============================================================
    # MODE INTERACTION ENTIRE STEP
    # ============================================================
        
    def mode_interaction_pre( 
        self,
        filters: list[KalmanFilter],
        x_fwd: np.ndarray,
        P_fwd: np.ndarray,
        x_premixed: np.ndarray,
        P_premixed: np.ndarray,
        x_rts: np.ndarray,
        P_rts: np.ndarray,
        PI: np.ndarray,
        comm_dim: float
    ):
        M = len(x_rts) # num of modes
        back_info_vectors = []
        back_info_matrices = []
        for i in range(M): 
            
            # DO ALL DATA FUSION W.R.T. EXTERNAL FRAME W/ PAD
            
            # So need to convert the forward pass and rts to external
            x_rts_e, P_rts_e = filters[i].to_external(x_rts[i], P_rts[i])
  
            P_rts_inv_i = np.linalg.inv(P_rts_e) # rts is already in internal dimension
            P_mixed_inv_i = np.linalg.inv(P_premixed[i])
            
            back_info_matrices.append(P_rts_inv_i - P_mixed_inv_i) # 7, one-step backward predicted information matrix
            back_info_vectors.append((P_rts_inv_i @ x_rts_e) - (P_mixed_inv_i @ x_premixed[i])) # 8, one-step backward predicted information vector      
         
        invertible = True
        for i in range(M):
            if np.linalg.matrix_rank(back_info_matrices[i]) != back_info_matrices[i].shape[0]:
                # This is checking if any of them arent invertible, in paper it does say really only care about i, but rather do this then dont have to renormalize
                invertible = False

        # Likelihood calculation still w.r.t. common frame
        mu_mix, likelihoods = self.get_mixing_weights(back_info_vectors, back_info_matrices, x_fwd, P_fwd, PI, invertible, comm_dim)
        return back_info_vectors, back_info_matrices, mu_mix, likelihoods, invertible
        
         
    def get_mixing_weights(self, Info_vectors, Info_matrices, x_fwd, P_fwd, PI, invertible, comm_dim: float):
        # ALWAYS W.R.T. COMMON FRAME
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
                    # Get likelihood WITH RESPECT TO Tcomm_dimHE TOTAL COMMON DIMENSION
                    Delta_ji = x_back_preds[i][0:comm_dim] - x_fwd[j][0:comm_dim]
                    D_ji = P_back_preds[i][0:comm_dim, 0:comm_dim] + P_fwd[j][0:comm_dim,0:comm_dim]
                    likelihood = gaussian_pdf(Delta_ji, np.zeros(len(Delta_ji)), D_ji)
                    likelihoods[j, i] = max(likelihood, 1e-32) # 10, Two-mode conditioned likelihood (clamped to 1e-25)
            print(f"Likelihoods: \n{likelihoods}")

            # Renormalize
            d = np.zeros(M)
            for j in range(M):
                d[j] = sum(PI[j, i] * likelihoods[j, i] for i in range(M))
            for i in range(M):
                for j in range(M):
                    mu_mix[i, j] = (PI[j, i] * likelihoods[j, i]) / d[j] # 11, Smoothed mixing probabilty
            return mu_mix, likelihoods
         
    def mode_interaction_1(self, filters, M, x_fwd, P_fwd, back_info_vectors, back_info_matrices, mu_mix, comm_dim: float, ext_dim: float):
        
        # DO EVERYTHING HERE IN EXTERNAL - ALL MIXING
        # THEN CONVERT BACK TO INTERNAL FOR FINAL MIXED FILTER EST
        
        # Compute the fwd info mat and vec, converting them to external frame
        fwd_info_matrices = []
        fwd_info_vectors = []
        for i in range(M):
            x_fwd_e, P_fwd_e = filters[i].to_external(x_fwd[i], P_fwd[i])
            fwd_info_matrices.append(np.linalg.inv(P_fwd_e))
            fwd_info_vectors.append(fwd_info_matrices[i] @ x_fwd_e)
            
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


    def mode_interaction_2(self, filters, M, x_fwd, P_fwd, back_state_vectors, back_cov_matrices, mu_mix, comm: float):
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
            # Convert to external for forward pass
            x_fwd_e, P_fwd_e = filters[j].to_external(x_fwd[j], P_fwd[j])
            
            P_pred_back_inv_j = np.linalg.inv(P_preds_back[j])
            P_hat_inv_j = np.linalg.inv(P_fwd_e)
            P_hats_back.append(np.linalg.inv(P_pred_back_inv_j + P_hat_inv_j)) # 7, Mode-conditioned smoothed covariance
            x_hats_back.append(P_hats_back[j] @ ((P_pred_back_inv_j @ x_preds_back[j]) + (P_hat_inv_j @ x_fwd_e))) # 6, Mode-conditioned smoothed mean

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
            # Check for divide by zero, nan, or inf in denominator or d vector
            if (
                den == 0.0
                or np.isnan(den)
                or np.isinf(den)
                or np.any(np.isnan(d))
                or np.any(np.isinf(d))
                or np.any(d == 0)
            ):
                return mu
            for j in range(M):
                mu_bck[j] = d[j] * mu[j] / den
            return mu_bck
        else:
            return mu

    def fuse_states_and_covs(self, filters, M, mus, xs, Ps):
        """
        Fuse states and covariances in external (common) coordinates.

        Args:
            M (int): Number of modes.
            mus (np.ndarray): Mode probabilities, shape (M,)
            xs (list[np.ndarray]): List of external state vectors, len M.
            Ps (list[np.ndarray]): List of external covariances, len M.
            filters (list[KalmanFilter]): List of filters (with to_external method).

        Returns:
            x_hat, P_hat: Fused external mean and covariance.
        """
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
        
        # COMMON DIMENSION THAT CAN BE PROPERLY MIXED AMONGST MODELS
        common = imm.common_dim

        # Allocate backward arrays as object arrays to allow variable-sized entries
        x_mode = np.empty((T, M), dtype=object)
        P_mode = np.empty((T, M), dtype=object)
        mu_s   = np.empty((T, M), dtype=object)

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
            "x_s": last['x_fuse'],
            "P_s": last['P_fuse'],
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
                
                print(f"At time step: {k}, time: {cache[k]["time"]}")
                
                cur = cache[k]
                nxt = cache[k + 1]
                PI = nxt["PI"]  # dt-scaled TPM from filtering, from cur to nxt.

                # ------------------------
                # MODE-MATCHED RTS
                # ------------------------
                x_rts = [None] * M
                P_rts = [None] * M

                for j in range(M):
                    x_rts[j], P_rts[j] = self._rts_step(
                        cur["x_filter"][j],
                        cur["P_filter"][j],
                        nxt["x_pred"][j],
                        nxt["P_pred"][j],
                        x_mode[k + 1, j],
                        P_mode[k + 1, j],
                        nxt["F"][j],
                    )

                # ------------------------
                # MODE INTERACTION
                # ------------------------
                back_info_vectors, back_info_matrices, mu_mix, likelihoods, invertible = self.mode_interaction_pre(
                    imm.filters,
                    cur['x_filter'],
                    cur['P_filter'],
                    nxt["x_premixed"],
                    nxt["P_premixed"],
                    x_rts,
                    P_rts,
                    PI,
                    common
                )

                if invertible:
                    # METHOD 2
                    back_state_vectors = []
                    back_cov_matrices = []
                    for j in range(M):
                        back_cov_matrices.append(np.linalg.inv(back_info_matrices[j]))
                        back_state_vectors.append(back_cov_matrices[j] @ back_info_vectors[j])
                    x_filt_smooth, P_filt_smooth = self.mode_interaction_2(imm.filters, M, cur['x_filter'], cur['P_filter'], back_state_vectors, back_cov_matrices, mu_mix, imm.common_dim)
                else:
                    # METHOD 1 
                    x_filt_smooth, P_filt_smooth = self.mode_interaction_1(imm.filters, M, cur['x_filter'], cur['P_filter'], back_info_vectors, back_info_matrices, mu_mix, imm.common_dim, imm.n)
        
                # ------------------------
                # MODE PROBABILITY SMOOTHING
                # ------------------------
                
                # ensure use PI from next cache, not curr
                mu_smooth = self.get_smoothed_pmodes(cur["mu"], PI, likelihoods, invertible)
                x_smooth, P_smooth = self.fuse_states_and_covs(imm.filters, M, mu_smooth, x_filt_smooth, P_filt_smooth)
            
                # Convert all of the filters to internal for saving
                for j in range(M):
                    x_filt_smooth[j], P_filt_smooth[j] = imm.filters[j].to_internal(x_filt_smooth[j], P_filt_smooth[j])

                # Save x_mode, P_mode
                x_mode[k] = x_filt_smooth
                P_mode[k] = P_filt_smooth
                
                # print(f"P_mode[1]: \n{P_filt_smooth[1]}")

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