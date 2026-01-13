import numpy as np
from utils import gaussian_pdf

def mode_matched_smoothing(next_fwd, bck):
    x_bars    = next_fwd["x_bars"]
    P_bars    = next_fwd["P_bars"]
    x_preds   = next_fwd["x_preds"]
    P_preds   = next_fwd["P_preds"]
    Cross_cov = next_fwd["C"]

    x_smooth_prev   = bck["x_hats"]
    P_smooth_prev   = bck["P_hats"]

    x_rts = []
    P_rts = []
    for i in range(len(x_bars)): # 1
        G = Cross_cov[i] @ np.linalg.inv(P_preds[i]) # 2, Smoother gain
        xb = x_bars[i] + G @ (x_smooth_prev[i] - x_preds[i]) # 3, Smoothed mixing mean
        Pb = P_bars[i] + G @ (P_smooth_prev[i] - P_preds[i]) @ G.T # 4, Smoothed mixing covariance
        x_rts.append(xb)
        P_rts.append(Pb)
    return x_rts, P_rts


def get_mixing_weights(Pi, Info_matrices, Info_vectors, x_hats, P_hats, invertible):
    if not invertible:
        return Pi, np.zeros_like(Pi)
    else:
        M = len(x_hats)
        x_back_preds = []
        P_back_preds = []
        for i in range(M):
            P_back_preds.append(np.linalg.inv(Info_matrices[i]))
            x_back_preds.append(P_back_preds[i] @ Info_vectors[i])

        mu_mix = np.zeros((M, M))
        likelihoods = np.zeros((M, M))
        # Extract Shared Statespace
        x_back_preds_shared = x_back_preds.copy()
        x_back_preds_shared[0] = x_back_preds[0][0:4]
        P_back_preds_shared = P_back_preds.copy()
        P_back_preds_shared[0] = P_back_preds[0][0:4, 0:4]

        x_hats_shared = x_hats.copy()
        x_hats_shared[0] = x_hats[0][0:4]
        P_hats_shared = P_hats.copy()
        P_hats_shared[0] = P_hats[0][0:4, 0:4]
        for i in range(M):
            for j in range(M):
                Delta_ji = x_back_preds_shared[i] - x_hats_shared[j]
                D_ji = P_back_preds_shared[i] + P_hats_shared[j]
                likelihoods[j, i] = gaussian_pdf(Delta_ji, np.zeros(len(Delta_ji)), D_ji) # 10, Two-mode conditioned likelihood

        d = np.zeros(M)
        for j in range(M):
            d[j] = sum(Pi[j, i] * likelihoods[j, i] for i in range(M))
        for i in range(M):
            for j in range(M):
                mu_mix[i, j] = (Pi[j, i] * likelihoods[j, i]) / d[j] # 11, Smoothed mixing probabilty
        return mu_mix, likelihoods


def mode_interaction_preliminary(M, Pi, fwd, next_fwd, x_rts, P_rts):
    x_hats = fwd["x_hats"]
    P_hats = fwd["P_hats"]
    x_bars = next_fwd["x_bars"]
    P_bars = next_fwd["P_bars"]

    back_info_matrices = []
    back_info_vectors = []
    fwd_info_matrices = []
    fwd_info_vectors = []
    for i in range(M): # 6
        P_rts_inv_i = np.linalg.inv(P_rts[i])
        P_bar_inv_i = np.linalg.inv(P_bars[i])
        # Forward information vectors / matrices
        fwd_info_matrices.append(np.linalg.inv(P_hats[i]))
        fwd_info_vectors.append(fwd_info_matrices[i] @ x_hats[i])
        # One-step backward predicted information vectors / matrices
        back_info_matrices.append(P_rts_inv_i - P_bar_inv_i) # 7, one-step backward predicted information matrix
        back_info_vectors.append((P_rts_inv_i @ x_rts[i]) - (P_bar_inv_i @ x_bars[i])) # 8, one-step backward predicted information vector

    invertible = True
    for i in range(M):
        if np.linalg.matrix_rank(back_info_matrices[i]) != back_info_matrices[i].shape[0]:
            invertible = False

    mu_mix, likelihoods = get_mixing_weights(Pi, back_info_matrices, back_info_vectors, x_hats, P_hats, invertible)
    return fwd_info_vectors, fwd_info_matrices, back_info_vectors, back_info_matrices, mu_mix, likelihoods, invertible


def mode_interaction_1(M, fwd_info_vectors, fwd_info_matrices, back_info_vectors, back_info_matrices, mu_mix):
    # 1. Combination
    n = back_info_vectors[0].shape[0]
    P_jis = np.zeros((M, M, n, n))
    x_hat_jis =  np.zeros((M, M, n))

    # Extract Shared Statespace
    fwd_info_vectors_shared = fwd_info_vectors.copy()
    fwd_info_vectors_shared[0] = fwd_info_vectors[0][0:4]
    fwd_info_matrices_shared = fwd_info_matrices.copy()
    fwd_info_matrices_shared[0] = fwd_info_matrices[0][0:4, 0:4]
    back_info_vectors_shared = back_info_vectors.copy()
    back_info_vectors_shared[0] = back_info_vectors[0][0:4]
    back_info_matrices_shared = back_info_matrices.copy()
    back_info_matrices_shared[0] = back_info_matrices[0][0:4, 0:4]

    # Two-mode conditioned smoothed estimates
    # CA - CA: Full fusion (9D)
    P_jis[0, 0, :, :] = np.linalg.inv(back_info_matrices[0] + fwd_info_matrices[0]) # 3, Two-mode conditioned smoothed covariance
    x_hat_jis[0, 0, :] = P_jis[0, 0, :, :] @ (back_info_vectors[0] + fwd_info_vectors[0]) # 4, Two-mode conditioned smoothed mean

    # CA - CV: Shared space and then re-insert accels
    P_jis_01 = np.linalg.inv(back_info_matrices_shared[1] + fwd_info_matrices_shared[0]) # 3, Two-mode conditioned smoothed covariance
    x_hat_jis_01 = P_jis_01 @ (back_info_vectors_shared[1] + fwd_info_vectors_shared[0]) # 4, Two-mode conditioned smoothed mean
    x_hat_jis[0, 1, :] = np.concatenate([x_hat_jis_01, x_hat_jis[0, 0, 4:6]])
    P_jis[0, 1, :, :] = np.block([
        [P_jis_01, P_jis[0, 0, 0:4, 4:6]],
        [P_jis[0, 0, 4:6, :]]])

    # CV - CA: Shared space and then re-insert accels
    P_jis_10 = np.linalg.inv(back_info_matrices_shared[0] + fwd_info_matrices_shared[1]) # 3, Two-mode conditioned smoothed covariance
    x_hat_jis_10 = P_jis_10 @ (back_info_vectors_shared[0] + fwd_info_vectors_shared[1]) # 4, Two-mode conditioned smoothed mean
    x_hat_jis[1, 0, :] = np.concatenate([x_hat_jis_10, x_hat_jis[0, 0, 4:6]])
    P_jis[1, 0, :, :] = np.block([
        [P_jis_10, P_jis[0, 0, 0:4, 4:6]],
        [P_jis[0, 0, 4:6, :]]])

    # CV - CV: Shared space and then re-insert accels
    P_jis_11 = np.linalg.inv(back_info_matrices_shared[1] + fwd_info_matrices_shared[1]) # 3, Two-mode conditioned smoothed covariance
    x_hat_jis_11 = P_jis_11 @ (back_info_vectors_shared[1] + fwd_info_vectors_shared[1]) # 4, Two-mode conditioned smoothed mean
    x_hat_jis[1, 1, :] = np.concatenate([x_hat_jis_11, x_hat_jis[0, 0, 4:6]])
    P_jis[1, 1, :, :] = np.block([
        [P_jis_11, P_jis[0, 0, 0:4, 4:6]],
        [P_jis[0, 0, 4:6, :]]])

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
    x_hats[1] = x_hats[1][0:4]
    P_hats[1] = P_hats[1][0:4, 0:4]
    return x_hats, P_hats


def mode_interaction_2(M, fwd, back_state_vectors, back_cov_matrices, mu_mix):
    x_hats = fwd["x_hats"]
    P_hats = fwd["P_hats"]

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
        P_hat_inv_j = np.linalg.inv(P_hats[j])
        P_hats_back.append(np.linalg.inv(P_pred_back_inv_j + P_hat_inv_j)) # 7, Mode-conditioned smoothed covariance
        x_hats_back.append(P_hats_back[j] @ ((P_pred_back_inv_j @ x_preds_back[j]) + (P_hat_inv_j @ x_hats[j]))) # 6, Mode-conditioned smoothed mean

    return x_hats_back, P_hats_back


def get_smoothed_pmodes(fwd, Pi, likelihoods, invertible):
    mu_fwd = fwd["mu"]
    M = mu_fwd.shape[0]
    if invertible:
        d = np.zeros(M)
        for j in range(M):
            for i in range(M):
                d[j] += Pi[j, i] * likelihoods[j, i]
        mu_bck = np.zeros(M)
        den = mu_fwd.T @ d
        for j in range(M):
            mu_bck[j] = d[j] * mu_fwd[j] / den
        return mu_bck
    else:
        return mu_fwd


def fuse_states_and_covs(M, mus, xs, Ps):
    # Extract Shared Statespace
    xs_shared = xs.copy()
    xs_shared[0] = xs[0][0:4]
    Ps_shared = Ps.copy()
    Ps_shared[0] = Ps[0][0:4, 0:4]

    x_hat = sum(mus[j] * xs_shared[j] for j in range(M))
    n = Ps_shared[0].shape[0]
    P_hat = np.zeros((n,n))

    for j in range(M):
        P_hat += mus[j] * (Ps_shared[j] + np.outer(xs_shared[j] - x_hat, xs_shared[j] - x_hat))

    x_hat = np.concatenate([x_hat, xs[0][4:6]])
    P_hat = np.block([
        [P_hat, Ps[0][0:4, 4:6]],
        [Ps[0][4:6, :]]])
    return x_hat, P_hat


def smooth_step(M, Pi, fwd, next_fwd, bck, mode_interaction):
    # Step 1. Mode-matched smoothing
    x_rts, P_rts = mode_matched_smoothing(next_fwd, bck)

    # Step 2. Mode interaction
    fwd_info_vectors, fwd_info_matrices, back_info_vectors, back_info_matrices, mu_mix, likelihoods, invertible = mode_interaction_preliminary(M, Pi, fwd, next_fwd, x_rts, P_rts)

    if mode_interaction == 1 or not invertible:
        x_hats_bck, P_hats_bck = mode_interaction_1(M, fwd_info_vectors, fwd_info_matrices, back_info_vectors, back_info_matrices, mu_mix)
    elif mode_interaction == 2:
        back_state_vectors = []
        back_cov_matrices = []
        for j in range(M):
            back_cov_matrices.append(np.linalg.inv(back_info_matrices[j]))
            back_state_vectors.append(back_cov_matrices[j] @ back_info_vectors[j])
        x_hats_bck, P_hats_bck = mode_interaction_2(M, fwd, back_state_vectors, back_cov_matrices, mu_mix)
    else:
        ValueError("Mode interaction must be 1 or 2")

    # Step 3. Smoother output
    mu_bck = get_smoothed_pmodes(fwd, Pi, likelihoods, invertible)
    x_hat_bck, P_hat_bck = fuse_states_and_covs(M, mu_bck, x_hats_bck, P_hats_bck)

    new_bck = {
        "x_hats": x_hats_bck,
        "P_hats": P_hats_bck,
        "mu": mu_bck
    }
    return new_bck

def smooth_ca_cv(forward, Pi, mode_interaction):
    T = len(forward)
    M = len(forward[0]["x_hats"])

    # Initialize with last step: smoothed = filtered
    backward = [{
        "x_hats": forward[-1]["x_hats"],
        "P_hats": forward[-1]["P_hats"],
        "mu": forward[-1]["mu"]
    }]

    # Backward pass
    for k in reversed(range(T - 1)):
        k_back = (T - 1) - k
        backward.append(smooth_step(M, Pi, forward[k], forward[k+1], backward[k_back - 1], mode_interaction))

    return backward[::-1]